#!/usr/bin/env python3
"""L4-L6: monotone onset alignment -> data/lyrics.approx.json.

The clock is the note onsets, not the presence envelope.  In sung Japanese and Chinese
one mora is one note, so the song's mora sequence should map onto the onset sequence
almost 1:1.  The DP allows exactly the two mismatches that really occur:

  * skip an onset  -- a breath, a backing vocal, an instrument leaking into the stem
                      (cheap for weak onsets, expensive for strong ones)
  * hold a mora    -- melisma: one note stretched over several morae (a held vowel)

Line boundaries are then pulled into the nearest silence between phrase runs, and the
result is written in the engine's schema (Word.syl = one span per glyph group, so a
2-mora kanji holds its glyph across two notes while a per-glyph wipe advances).

Run:  .venv/bin/python analysis/align_lyrics.py [--stems] [--no-plots]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))  # legacy/: shared lib lives in analysis/

import numpy as np

import common
import lyric_units

HOLD = 0.9            # cost of stretching one note over one more mora (melisma)
SKIP = 0.6            # cost of dropping an onset, scaled by (1 - strength)
WEAK_MATCH = 0.25     # cost of matching a mora to a weak onset

FPS = common.FPS      # envelopes are 100 fps, so frame == centisecond


def gated_onsets(ot, os_, runs, margin=0.25):
    """Keep only onsets inside the singable interval the vocal gate defines.

    The stem is not silent where there is no singing -- reverb tails and drum bleed survive,
    so at a low threshold the 60 s instrumental intro and the fade after the last note still
    yield onsets, and anchoring lyrics to those was what stretched the first and last lines.

    The gate is the *interval* from the first to the last phrase run (plus `margin`), not the
    union of the runs: the hysteresis gate dips below its off threshold between syllables, so
    intersecting with the runs themselves throws away real notes inside phrases (dropping to
    0.5 onsets per mora and forcing every word to collapse onto one note)."""
    if not len(runs) or not len(ot):
        return ot, os_
    lo, hi = float(runs[0][0]) - margin, float(runs[-1][1]) + margin
    keep = (ot >= lo) & (ot <= hi)
    return ot[keep], os_[keep]


def load_features(src: str) -> dict:
    z = np.load(common.WORK / ("features_%s.npz" % src))
    return {k: z[k] for k in z.files}


def unit_starts(units, slot_unit, slot_onset, ot) -> np.ndarray:
    """Note start per unit, forward-filled (a unit whose slots were all held shares the
    previous note), so the result is monotone non-decreasing by construction."""
    n = len(units)
    starts = np.full(n, np.nan, dtype=np.float64)
    for s_i, u in enumerate(slot_unit):
        if slot_onset[s_i] >= 0 and np.isnan(starts[u]):
            starts[u] = ot[slot_onset[s_i]]
    prev = float(ot[0]) if len(ot) else 0.0
    filled = 0
    for u in range(n):
        if np.isnan(starts[u]):
            starts[u] = prev
            filled += 1
        else:
            prev = starts[u]
    return starts, filled


def line_spans(units, starts, tail) -> list[list[float]]:
    """[start, end, last-note start] per line; lines are contiguous and monotone."""
    nl = max(u["line"] for u in units) + 1
    lo = [None] * nl
    hi = [None] * nl
    last = [None] * nl
    for u_i, u in enumerate(units):
        li = u["line"]
        if lo[li] is None:
            lo[li] = starts[u_i]
        hi[li] = starts[u_i + 1] if u_i + 1 < len(units) else starts[u_i] + tail
        last[li] = starts[u_i]
    return [[float(lo[i]), float(hi[i]), float(last[i])] for i in range(nl)]


def snap_lines(spans, runs, tol=0.8):
    """Pull line boundaries into the nearest silence between phrase runs.

    A boundary may not be pulled in front of the previous line's last note: doing so makes
    the line's final glyph collapse into a few milliseconds before the note it belongs to
    (the clamp has nowhere else to put it).  When no silence is available in that window the
    DP's own boundary stands, which is the next line's first onset.
    """
    bounds = [s[0] for s in spans] + [spans[-1][1]]
    gaps = [0.5 * (runs[i][1] + runs[i + 1][0]) for i in range(len(runs) - 1)]
    snapped = [False] * len(bounds)
    for k in range(1, len(bounds) - 1):
        if not gaps:
            break
        floor = spans[k - 1][2] + 0.02
        cands = [g for g in gaps if g >= floor]
        if not cands:
            continue
        near = min(cands, key=lambda g: abs(g - bounds[k]))
        if abs(near - bounds[k]) <= tol:
            bounds[k], snapped[k] = float(near), True
    for k in range(1, len(bounds)):
        if bounds[k] < bounds[k - 1] + 0.20:
            bounds[k] = bounds[k - 1] + 0.20
    return bounds, snapped


def act_end(t, act, thr=0.05, gap=0.12):
    """When the voice stops after t: the first frame that begins >= `gap` s of silence.

    The cap must come from the presence envelope itself, not from the stored phrase runs.
    Those are hysteresis-gated (0.18 on / 0.10 off) and merged and split for the phrase map,
    so they are trimmed short and fragmented: capping on them cut a note down to 30 ms in the
    pilot render and left a 2 s hole inside a sung phrase (L0, 63.9-65.9 s) because the run
    ended while the voice kept going.  `act` is the same normalised envelope the coverage
    metric and the onset gate use, so the three agree by construction.
    """
    i, n, quiet = max(0, int(round(t * FPS))), len(act), 0
    while i < n:
        quiet = quiet + 1 if act[i] <= thr else 0
        if quiet >= gap * FPS:
            return (i - quiet + 1) / FPS
        i += 1
    return float("inf")


def line_conf(t0, t1, act, ot, morae):
    """How much the audio corroborates a line's span, 0..1.

    Two things have to hold for the timing to be worth trusting: the voice is actually present
    across the span (rather than the line being stretched over a silence), and the number of
    notes inside the span is about the number of morae the text asks for.  A line spread over a
    stop-start section scores low, which is the honest answer -- its boundary is only as good as
    the monotonicity of the DP there.
    """
    a, b = int(round(t0 * FPS)), int(round(t1 * FPS))
    seg = act[a:b] if b > a else np.zeros(0, dtype=np.float32)
    cov = float((seg > 0.05).mean()) if seg.size else 0.0
    notes = int(((ot >= t0) & (ot <= t1)).sum()) if len(ot) else 0
    ratio = notes / max(1.0, float(morae))
    return round(min(0.6 * cov + 0.4 * min(ratio, 1.0), 0.5 + 0.5 * cov), 2), cov, notes


def build(lines, units, starts, bounds, act, ot):
    """Assign final times: units clamped inside their line's span, words from units."""
    out, rep = [], []
    for li, lu in enumerate(lines):
        t0, t1 = float(bounds[li]), float(bounds[li + 1])
        idx = [i for i, u in enumerate(units) if u["line"] == li]
        # units that start before the line (or after it) are clamped into the span
        clamped = 0
        us = []
        for i in idx:
            t = min(max(starts[i], t0 + 0.005), t1 - 0.005)
            if abs(t - starts[i]) > 0.02:
                clamped += 1
            us.append(t)
        us[-1] = min(us[-1], t1 - 0.02)
        # Monotone, but a unit that would start before its predecessor *shares* that note:
        # the melisma the DP held (a 2-mora kanji sung on one note) is real, and its glyph
        # group should hold the note rather than be pushed 10 ms apart.
        prev = us[0]
        for k in range(1, len(us)):
            prev = us[k] = max(us[k], prev)
        # Split each note evenly across the units that share it, so every glyph still gets
        # screen time; spans are clamped to the note so the group ends exactly on it.
        spans, k = [], 0
        while k < len(us):
            j = k
            while j + 1 < len(us) and us[j + 1] <= us[k] + 1e-9:
                j += 1
            end = us[j + 1] if j + 1 < len(us) else t1
            # A note ends when the voice stops, not at the next onset: between the bursts of a
            # stop-start section the next onset is 6 s away, which would otherwise stretch the
            # note -- and the whole line -- across the silence.
            end = min(end, act_end(us[k], act))
            if end <= us[k]:
                end = us[k] + 0.02
            # Divide the note exactly, so the last slice ends on `end` and every slice has a
            # width: a `max(step, 0.005)` floor instead walks the split past `end` and leaves
            # the last glyph a zero-length span.  A note too short to divide is shared whole.
            n, room = j - k + 1, end - us[k]
            if room < 0.001 * n:
                spans.extend([(us[k], end)] * n)
            else:
                for m in range(k, j + 1):
                    spans.append((us[k] + room * (m - k) / n, us[k] + room * (m - k + 1) / n))
            k = j + 1
        conf, cov, notes = line_conf(t0, t1, act, ot, sum(units[i]["morae"] for i in idx))
        words = []
        for wi, (tok, uu) in enumerate(zip(lu.tokens, lu.units)):
            sub = [spans[k] for k, gi in enumerate(idx) if units[gi]["word"] == wi]
            syl = [[round(a, 3), round(b, 3)] for a, b in sub if b > a]
            d = {"w": tok, "start": round(sub[0][0], 3), "end": round(sub[-1][1], 3),
                 "conf": conf}
            if len(syl) > 1:
                d["syl"] = syl
            words.append(d)
        out.append({"i": li, "text": lu.text, "start": round(t0, 3), "end": round(t1, 3),
                    "words": words})
        rep.append({"i": li, "start": round(t0, 3), "end": round(t1, 3),
                    "dur": round(t1 - t0, 3), "units": lu.n_units, "clamped": clamped,
                    "conf": conf, "cov": round(cov, 3), "notes": notes})
    return out, rep


def report(rep, counts, O, S, filled, snapped, bounds, ot, tail) -> str:
    per = [abs(b - float(ot[np.argmin(np.abs(ot - b))])) for b in bounds[1:-1]] if len(ot) else []
    buckets = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 10.0]
    hist = {f"<{buckets[i + 1]:g}s": 0 for i in range(len(buckets) - 1)}
    for d in per:
        for i in range(len(buckets) - 1):
            if d < buckets[i + 1]:
                hist[f"<{buckets[i + 1]:g}s"] += 1
                break
    o = ["L4-L6 onset alignment",
         "  mora slots        %d" % S,
         "  onsets            %d   -> surplus %.2fx (1.0 means one note per mora)" % (O, O / max(1, S)),
         "  DP transitions    matched %d   held (melisma) %d   skipped %d"
         % (counts["match"], counts["hold"], counts["skip"]),
         "  units sharing a note (no own onset, forward-filled): %d" % filled,
         "  line boundaries snapped into a silence: %d of %d" % (sum(1 for s in snapped if s), len(snapped)),
         "  median note length %.3f s   closing tail %.3f s" % (tail, tail),
         "  line boundaries vs nearest onset: " + "  ".join("%s:%d" % (k, v) for k, v in hist.items()),
         "",
         "  i   start     end    dur  units  notes   cov  conf  clamp",
         ]
    for r in rep:
        o.append("  %3d %8.3f %8.3f %6.2f %5d %6d %5.2f %5.2f  %5d"
                 % (r["i"], r["start"], r["end"], r["dur"], r["units"], r["notes"],
                    r["cov"], r["conf"], r["clamped"]))
    durs = [r["dur"] for r in rep]
    confs = [r["conf"] for r in rep]
    o.append("")
    o.append("  line durations: median %.2f s  p90 %.2f s  min %.2f  max %.2f"
             % (float(np.median(durs)), float(np.percentile(durs, 90)), min(durs), max(durs)))
    o.append("  confidence: median %.2f   lines under 0.5: %d"
             % (float(np.median(confs)), sum(1 for c in confs if c < 0.5)))
    low = sorted(rep, key=lambda r: (r["conf"], -r["dur"]))[:6]
    o.append("  least corroborated (span, coverage, notes): "
             + "  ".join("L%d %.1fs/%.0f%%/%d" % (r["i"], r["dur"], 100 * r["cov"], r["notes"])
                         for r in low))
    return "\n".join(o)


def main(argv: list[str]) -> int:
    src = "vocal" if "--vocal" in argv else ("stem" if "--stems" in argv else "mix")
    dest = common.ROOT / (argv[argv.index("--out") + 1] if "--out" in argv
                         else "data/lyrics.approx.json")
    f = load_features(src)
    runs = f["runs"]
    ot_all, os_all = f["onsets"][0].astype(np.float64), f["onsets"][1].astype(np.float64)
    ot, os_ = gated_onsets(ot_all, os_all, runs)
    lines, unresolved, ja, zh = lyric_units.parse()
    units, _ = flatten_units(lines)
    slot_unit = mora_slots(units)
    slot_onset, counts, cost = align_slots(slot_unit, ot, os_)
    starts, filled = unit_starts(units, slot_unit, slot_onset, ot)
    tail = float(np.median(np.diff(ot))) * 2.0 if len(ot) > 2 else 0.4
    spans = line_spans(units, starts, tail)
    bounds, snapped = snap_lines(spans, runs)
    out, rep = build(lines, units, starts, bounds, f["vocal"], ot)

    txt = report(rep, counts, len(ot), len(slot_unit), filled, snapped, bounds, ot, tail)
    (common.QA / ("l4_report_%s.txt" % src)).write_text(txt + "\n", encoding="utf-8")
    print(txt[:3500])

    doc = {"lines": out, "extras": [],
           "notes": ("onset-aligned lyric timing (%s features).  Monotone DP over the song's "
                     "mora sequence and the note onsets inside the vocal-active gate (%d of "
                     "%d detected; the rest is stem bleed in the instrumental intro and the "
                     "fade), so every unit boundary lands on a real note onset: %d morae, %d "
                     "matched, %d held (melisma), %d skipped; %d units share a note with the "
                     "previous one; line boundaries pulled into the nearest silence within "
                     "0.8 s and clamped inside their line, and every unit's span cut where the "
                     "voice stops, so a note in a stop-start section ends with its burst instead "
                     "of running to the next onset seconds later.  conf = 0.6 x (fraction of the "
                     "line's span where the voice is present) + 0.4 x (notes in the span / morae), "
                     "capped by that coverage: a line stretched across a silence scores low.  "
                     "Weights: Japanese "
                     "kanji 1..3 (from onsets), Chinese one per glyph, kana one per glyph, "
                     "Latin one per syllable."
                     % (src, len(ot), len(ot_all), len(slot_unit), counts["match"],
                        counts["hold"], counts["skip"], filled)),
           "source": "analysis/align_lyrics.py",
           "method": {"features": src, "morae": len(slot_unit), "onsets": len(ot),
                      "onsets_detected": len(ot_all), "matched": counts["match"],
                      "held": counts["hold"], "skipped": counts["skip"],
                      "conf_median": float(np.median([r["conf"] for r in rep])),
                      "conf_low": sum(1 for r in rep if r["conf"] < 0.5),
                      "cost": round(cost, 2)}}
    doc["method"]["stem_delay_ms"] = round(1000.0 * float(f.get("offset", 0.0)), 1)
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\nwrote %s  (%d lines)" % (dest, len(out)))
    print("first: " + " | ".join("%d %.2f-%.2f" % (d["i"], d["start"], d["end"]) for d in out[:4]))
    print("last : " + " | ".join("%d %.2f-%.2f" % (d["i"], d["start"], d["end"]) for d in out[-4:]))

    if "--no-plots" not in argv:
        import analyze
        feat = {"nf": int(f["nf"]), "runs": [tuple(r) for r in runs],
                "onsets": f["onsets"].astype(np.float32),
                "env": {k: f[k] for k in ("rms", "low", "mid", "high", "vocal", "vocal_raw")}}
        print("plots: %d written" % analyze.plots(feat, rep, "align_" + src))
    return 0


def flatten_units(lines):
    """-> (units, line_of_unit); Latin words are split into one unit per syllable.

    A CJK/kana unit keeps its mora count (a 2-mora kanji holds its glyph over two
    notes); a Latin word becomes one unit per vowel group, all carrying the same text,
    because Lyrics.lineCharProgress sweeps the word's characters across its `syl` spans.
    """
    units, line_of = [], []
    for li, lu in enumerate(lines):
        for wi, us in enumerate(lu.units):
            for u in us:
                n = max(1, int(round(u.morae)))
                if u.cls == "latin" and n > 1:
                    for m in range(n):
                        units.append({"line": li, "word": wi, "text": u.text, "morae": 1,
                                      "lang": u.lang, "tentative": True})
                        line_of.append(li)
                else:
                    units.append({"line": li, "word": wi, "text": u.text, "morae": n,
                                  "lang": u.lang, "tentative": u.tentative})
                    line_of.append(li)
    return units, line_of


def mora_slots(units) -> np.ndarray:
    """Index of the unit each mora slot belongs to (len == total morae)."""
    return np.repeat(np.arange(len(units)), [u["morae"] for u in units])


def align_slots(slot_unit: np.ndarray, ot: np.ndarray, os_: np.ndarray,
                hold=HOLD, skip=SKIP, weak=WEAK_MATCH, band=0.2):
    """Monotone DP over (mora slot, onset), restricted to a Sakoe-Chiba band.

    Transitions into (i, j):
      match  (i-1, j-1) + weak * (1 - strength[j-1])
      hold   (i-1, j)   + hold              (one note over several morae)
      skip   (i,   j-1) + skip * (1 - strength[j-1])

    The end state minimises dp[S, j] plus the cost of dropping every onset after j, so
    trailing onsets in a fade or a noisy tail are skipped rather than forced onto the
    last mora.
    """
    S, O = len(slot_unit), len(ot)
    w = max(60, int(band * O))
    INF = np.float32(1e9)
    dp = np.full((S + 1, O + 1), INF, dtype=np.float32)
    back = np.zeros((S + 1, O + 1), dtype=np.uint8)      # 1 match, 2 hold, 3 skip
    dp[0, 0] = 0.0
    sk = (skip * (1.0 - os_)).astype(np.float32)
    wa = (weak * (1.0 - os_)).astype(np.float32)
    for i in range(1, S + 1):
        c = int(round(i * O / S))
        dpi, bpi, dpp = dp[i], back[i], dp[i - 1]
        for j in range(max(1, c - w), min(O, c + w) + 1):
            best, how = dpp[j - 1] + wa[j - 1], 1
            v = dpp[j] + hold
            if v < best:
                best, how = v, 2
            v = dpi[j - 1] + sk[j - 1]
            if v < best:
                best, how = v, 3
            dpi[j], bpi[j] = best, how

    cum = np.concatenate([[0.0], np.cumsum(sk)]).astype(np.float32)
    ends = dp[S, :] + (cum[O] - cum[: O + 1])
    j = int(np.argmin(ends))
    cost = float(ends[j])
    i = S
    slot_onset = np.full(S, -1, dtype=np.int32)
    counts = {"match": 0, "hold": 0, "skip": 0}
    while i > 0 or j > 0:
        how = back[i, j]
        if i > 0 and j > 0 and how == 1:
            slot_onset[i - 1] = j - 1
            counts["match"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and how == 2:
            counts["hold"] += 1
            i -= 1
        elif j > 0 and how == 3:
            counts["skip"] += 1
            j -= 1
        else:
            counts["skip"] += max(0, j)
            break
    return slot_onset, counts, cost


if __name__ == "__main__":
    sys.exit(main(sys.argv))
