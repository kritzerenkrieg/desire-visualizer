#!/usr/bin/env python3
"""L7 pilot: the first syllables over the audio -- 60 s to 120 s.

Nothing upstream renders video on this host (no node/bun), so the pilot is built the way the
rest of analysis/ works: matplotlib draws, tools/ff/ffmpeg muxes.  It exists to judge the
syllable timings of data/lyrics.approx.json by ear and eye together, before any scene work.

  out/pilot_<a>_<b>.mp4        20 fps: the line with its wipe, the syllable timeline, the
                               audio muxed from audio/source.mp3 at exactly `--start`
  out/pilot_<a>_<b>_strip.png  static score strip: every unit block, the vocal presence, the
                               note onsets, the line bounds and the chop gaps

The wipe is the engine's own model -- Lyrics.lineCharProgress over Lyrics.wordProgress, which
is piecewise across Word.syl -- so what this shows is what pdoom-video/app will draw for the
same data.  Words are monotone and non-overlapping, and wordProgress is non-decreasing, so the
sung region is always a prefix of the line: one clip rectangle is exact, not an approximation.

common.OFFSET_MS is 0.0 (analysis time == mux time, see common.py), so the audio segment
starts at `--start` with no correction.

Run:  .venv/bin/python analysis/pilot.py                        # 60-120 s, mp4 + strip
      .venv/bin/python analysis/pilot.py --start 174 --end 216  # the chopped section
      .venv/bin/python analysis/pilot.py --src vocal --data analysis/work/lyrics_vocal.json
      .venv/bin/python analysis/pilot.py --no-video             # strip only

--src picks which features_<src>.npz the presence curve and the onsets come from (mix / stem /
vocal) and --data an alignment other than data/lyrics.approx.json, so a candidate pass can be
watched against the adopted one without touching it.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

import common

import lyric_units

FFMPEG = common.TOOLS / "ff" / "ffmpeg"
FONTS = (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",     # covers kana + kanji
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    str(common.WORK / "fonts" / "NotoSansJP-Regular.otf"),           # fetched when the host has none
    str(common.ROOT / "reference" / "pdoom-video" / "app" / "public" / "fonts" / "Cormorant-400.ttf"),
)
CHOP_GAP = 0.35            # a silence longer than this between units is an edit, not phrasing
ACT_FLOOR = 0.05           # vocal presence above which the voice is considered present
BG, FG, DIM = "#0b1020", "#e2e8f0", "#5b6478"
LIT, SUNGC, PEND, CHOP, CUR = "#ffffff", "#38bdf8", "#3f4a60", "#7f1d1d", "#f59e0b"
UNDIM = "#39415a"          # the unsung part of the current line (must read as clearly darker)


def font_candidates() -> list[str]:
    """The app's own faces first, then everything the host has installed."""
    out = [p for p in FONTS if common.Path(p).exists()]
    for root in ("/usr/share/fonts", "/usr/local/share/fonts"):
        if common.Path(root).exists():
            for ext in ("*.ttf", "*.otf", "*.ttc"):
                out += [str(p) for p in sorted(common.Path(root).rglob(ext))]
    return list(dict.fromkeys(out))


def pick_font(path: str | None = None) -> tuple[str, str]:
    """(file, family) of the face that covers the most of the lyrics.

    Measured on this host the best face is CJK-only -- Droid Sans Fallback has every kana and
    kanji and no Latin at all -- so the caller draws with `[family, "DejaVu Sans"]` and lets
    matplotlib fall back per glyph: Japanese from the CJK face, 'faith of' / 'Poker Face' from
    DejaVu.  A missing Japanese glyph is fatal to a lyric pilot, a missing Latin one is not,
    so the two are scored separately instead of trusting any one font.
    """
    from fontTools.ttLib import TTFont
    from matplotlib import font_manager as fm
    need = {c for c in "".join(lu.text for lu in lyric_units.parse()[0]) if c.strip()}
    latin, jp = {c for c in need if ord(c) < 0x2E80}, {c for c in need if ord(c) >= 0x2E80}
    best = None
    for cand in ([path] if path else font_candidates()):
        try:
            t = TTFont(cand, fontNumber=0, lazy=True)
            cmap = set(t.getBestCmap())
        except Exception:
            continue
        score = (sum(1 for c in jp if ord(c) not in cmap),
                 sum(1 for c in latin if ord(c) not in cmap))
        if best is None or score < best[0]:
            try:
                fam = t["name"].getDebugName(1) or ""
            except Exception:
                fam = ""
            best = (score, cand, fam)
    if best is None:
        raise SystemExit("no usable font found -- pass --font")
    (mj, ml), fname, fam = best
    fm.fontManager.addfont(fname)
    print("font %s\n  family %r   Japanese %d/%d   Latin left to DejaVu %d"
          % (fname, fam, len(jp) - mj, len(jp), ml))
    return fname, fam


def progress(words: list[dict], t: float) -> float:
    """Chars of the line sung at t -- a port of Lyrics.lineCharProgress."""
    n, chars = 0.0, 0.0
    for w in words:
        p = word_progress(w, t)
        chars += p * len(w["w"])
        if p < 1:
            break
        chars += 1                                  # the space between words
        n += 1
    return chars


def word_progress(w: dict, t: float) -> float:
    """Sung progress of a word -- a port of Lyrics.wordProgress (piecewise across syl)."""
    if t <= w["start"]:
        return 0.0
    if t >= w["end"]:
        return 1.0
    syl = w.get("syl") or []
    if len(syl) > 1:
        n = len(syl)
        # spans are {name,start,end} objects in the current engine schema, (a, b) in older ones
        spans = [(float(s["start"]), float(s["end"])) if isinstance(s, dict)
                 else (float(s[0]), float(s[1])) for s in syl]
        for i, (a, b) in enumerate(spans):
            if t < a:
                return i / n
            if t < b:
                return (i + (t - a) / max(1e-3, b - a)) / n
        return 1.0
    return (t - w["start"]) / max(1e-3, w["end"] - w["start"])


def units_of(doc: dict) -> list[dict]:
    """One entry per engine unit: a Word.syl span, or the word itself when it has none.

    Times come only from the JSON (that is what the engine reads); the glyph text comes from
    lyric_units via the aligner's own split, so the labels match the lyric source.
    """
    lines, _, _, _ = lyric_units.parse()
    flat = lyric_units.flatten_units(lines)[0]
    out, k = [], 0
    for ln in doc["lines"]:
        for wi, w in enumerate(ln["words"]):
            syl = [(float(s["start"]), float(s["end"])) if isinstance(s, dict)
                   else (float(s[0]), float(s[1])) for s in (w.get("syl") or [])]
            labels = list(w.get("sung") or [])          # one label per emitted span
            if not syl:                                  # no spans = one unit: the word itself
                syl = [(float(w["start"]), float(w["end"]))]
                labels = labels or [w["w"]]
            if len(labels) != len(syl):
                us = [u for u in flat if u["line"] == ln["i"] and u["word"] == wi]
                labels = ([u["text"] for u in us] if len(us) == len(syl)
                          else [w["w"]] * len(syl))
            for (a, b), txt in zip(syl, labels):
                out.append({"i": k, "line": ln["i"], "word": w["w"], "text": txt,
                            "start": a, "end": b, "conf": w.get("conf")})
                k += 1
    return out


def act_runs(act: np.ndarray, thr=ACT_FLOOR, min_s=0.08) -> list[tuple[float, float]]:
    """Spans where the voice is present; the complement is a silence or a production edit."""
    s = act > thr
    out, i = [], 0
    while i < len(s):
        if s[i]:
            j = i
            while j + 1 < len(s) and s[j + 1]:
                j += 1
            if (j - i + 1) / common.FPS >= min_s:
                out.append((i / common.FPS, (j + 1) / common.FPS))
            i = j + 1
        else:
            i += 1
    return out


def unit_gaps(units: list[dict], gap=CHOP_GAP) -> list[tuple[float, float, int]]:
    """Silences between consecutive units of a line -- the chops, with their line number."""
    return [(float(a["end"]), float(b["start"]), a["line"])
            for a, b in zip(units, units[1:])
            if a["line"] == b["line"] and b["start"] - a["end"] >= gap]


def load(src: str = "stem", data: str | None = None) -> dict:
    """The alignment to judge, plus the features it was aligned against.

    `src` picks features_<src>.npz (mix / stem / vocal), `data` an alignment other than the one
    the engine currently reads, so a candidate pass can be rendered beside the adopted one.
    """
    doc = json.loads((common.DATA / "lyrics.approx.json").read_text(encoding="utf-8")
                     if not data else str(common.ROOT / data))
    z = np.load(common.WORK / ("features_%s.npz" % src))
    return {"doc": doc, "act": z["vocal"].astype(np.float64),
            "rms": z["rms"].astype(np.float64), "onsets": z["onsets"][0].astype(np.float64),
            "units": units_of(doc)}


def in_window(items, start, end):
    return [u for u in items if u["start"] < end and u["end"] > start]


def _labels(ax, sel, family, size=8, dy=(0.45, -0.55)):
    """Glyph labels, alternating above/below a block so narrow ones do not collide."""
    from matplotlib.font_manager import FontProperties
    jp = FontProperties(family=[family, "DejaVu Sans"], size=size)
    for k, u in enumerate(sel):
        ax.text(0.5 * (u["start"] + u["end"]), dy[k % 2], u["text"], ha="center", va="center",
                fontproperties=jp, color=FG if u["end"] - u["start"] > 0.10 else DIM, clip_on=True)


def _chops(ax, gaps, y0, y1):
    for a, b, _ in gaps:
        ax.axvspan(a, b, y0, y1, color=CHOP, alpha=0.45, lw=0)


def strip(start: float, end: float, D: dict, font: tuple[str, str], path) -> str:
    """Static score strip: presence, onsets, every unit block, the chops, the mix level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    fname, family = font
    doc, act, gaps = D["doc"], D["act"], unit_gaps(D["units"])
    units = in_window(D["units"], start, end)
    lines = [ln for ln in doc["lines"] if ln["start"] < end and ln["end"] > start]
    jp = FontProperties(family=[family, "DejaVu Sans"], size=10)
    t = np.arange(len(act)) / common.FPS
    m = (t >= start) & (t <= end)
    nrow = len(lines)
    fig = plt.figure(figsize=(34, 3.2 + 1.15 * nrow), dpi=70, facecolor="white")
    gs = fig.add_gridspec(nrow + 2, 1, height_ratios=[2.1] + [1.15] * nrow + [1.0], hspace=0.35,
                          left=0.088, right=0.985, top=0.94, bottom=0.05)

    ax = fig.add_subplot(gs[0])
    ax.fill_between(t[m], -0.02, act[m], color="#7c3aed", alpha=0.30, lw=0)
    ax.plot(t[m], act[m], color="#6d28d9", lw=0.7)
    ax.vlines(D["onsets"][(D["onsets"] >= start) & (D["onsets"] <= end)], -0.02, 0.30,
              color="k", lw=0.6, alpha=0.55)
    for a, b, _ in gaps:
        if b > start and a < end:
            ax.axvspan(a, b, color=CHOP, alpha=0.28, lw=0)
            if b - a >= 1.0:
                ax.text(0.5 * (a + b), 0.85, "chop %.1f s" % (b - a), ha="center",
                        color="#7f1d1d", fontsize=7)
    for ln in lines:
        ax.axvline(ln["start"], color=CUR, lw=1.0)
        ax.text(ln["start"], 1.08, "L%d conf %.2f" % (ln["i"], ln["words"][0]["conf"]),
                color=CUR, fontsize=8, ha="left", va="bottom")
    ax.set_xlim(start, end)
    ax.set_ylim(-0.05, 1.32)
    ax.set_ylabel("vocal stem\npresence\n+ onsets")
    ax.set_title("pilot %.1f-%.1f s   %d units   %d lines   %d chops >= %.2f s   "
                 "(blue bars = the syllable the aligner claims)"
                 % (start, end, len(units), len(lines),
                    len([g for g in gaps if g[1] > start and g[0] < end]), CHOP_GAP),
                 fontsize=11, loc="left")

    for r, ln in enumerate(lines):
        ax = fig.add_subplot(gs[r + 1])
        sel = [u for u in units if u["line"] == ln["i"]]
        _chops(ax, [g for g in gaps if g[1] > ln["start"] - 1 and g[0] < ln["end"] + 1], 0.2, 0.8)
        for k, u in enumerate(sel):
            ax.barh(0, u["end"] - u["start"], left=u["start"], height=0.62, color=SUNGC,
                    edgecolor="#0f172a", lw=0.5)
            if k and sel[k - 1]["word"] != u["word"]:
                ax.axvline(u["start"], color="#0f172a", lw=0.9)
        _labels(ax, sel, family)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlim(start, end)
        ax.set_yticks([])
        ax.text(0.004, 0.86, "L%d" % ln["i"], transform=ax.transAxes, fontsize=8, color=CUR,
                va="top", ha="left")
        ax.set_ylabel(ln["text"][:12], fontproperties=jp, fontsize=8, rotation=0, ha="right",
                      va="center", labelpad=18)
        ax.tick_params(labelleft=False)
        if r < nrow - 1:
            ax.tick_params(labelbottom=False)
    ax.set_xlabel("song time (s)")
    ax = fig.add_subplot(gs[nrow + 1])
    ax.fill_between(t[m], 0, D["rms"][m], color="#0f172a", lw=0)
    ax.set_xlim(start, end)
    ax.set_ylabel("mix rms")
    fig.savefig(path, dpi=70, facecolor="white")
    plt.close(fig)
    return str(path)


def video(start: float, end: float, D: dict, font: tuple[str, str], path, fps=20, w=1280,
          h=720) -> str:
    """Render the segment frame by frame and mux the audio with ffmpeg.

    Only the moving pieces (the wipe, the playheads, the active block, the readouts) are
    redrawn per frame and blitted over a saved background: a full redraw of ~200 static
    artists per frame at 20 fps would take a quarter hour, this takes under a minute.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    from matplotlib.textpath import TextToPath

    fname, family = font
    doc, rms, onsets = D["doc"], D["rms"], D["onsets"]
    units, gaps = in_window(D["units"], start, end), unit_gaps(D["units"])
    lines = [ln for ln in doc["lines"] if ln["start"] < end and ln["end"] > start]
    jp = FontProperties(family=[family, "DejaVu Sans"], size=25)
    jp_s = FontProperties(family=[family, "DejaVu Sans"], size=14)
    mono = FontProperties(family="DejaVu Sans Mono", size=10)
    mono_jp = FontProperties(family=[family, "DejaVu Sans Mono"], size=10)
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.7, 0.55], hspace=0.16,
                          left=0.045, right=0.985, top=0.96, bottom=0.04)
    axL, axT, axI = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for ax in (axL, axT, axI):
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#26314a")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)

    # lyric panel: the neighbours are static context; the current line is drawn twice -- dim
    # underneath and lit on top -- with the lit copy clipped to the sung prefix of the line
    prev_t = axL.text(0.06, 0.90, "", fontproperties=jp_s, color=DIM, va="center", ha="left",
                      animated=True)
    next_t = axL.text(0.06, 0.10, "", fontproperties=jp_s, color=DIM, va="center", ha="left",
                      animated=True)
    dim_t = axL.text(0.06, 0.50, "", fontproperties=jp, color=UNDIM, va="center", ha="left",
                     animated=True)
    lit_t = axL.text(0.06, 0.50, "", fontproperties=jp, color=LIT, va="center", ha="left",
                     animated=True)
    clip = Rectangle((0.06, 0.30), 0.001, 0.42, transform=axL.transAxes, facecolor="none",
                     edgecolor="none")
    axL.add_patch(clip)
    lit_t.set_clip_path(clip)
    wipe = axL.axvline(0.06, ymin=0.30, ymax=0.72, color=CUR, lw=1.4, animated=True)

    # timeline panel: one row per line, one block per unit, the chops shaded
    axT.set_xlim(start, end)
    axT.set_ylim(len(lines) - 0.55, -0.7)
    for r, ln in enumerate(lines):
        for u in units:
            if u["line"] == ln["i"]:
                axT.barh(r, u["end"] - u["start"], left=u["start"], height=0.6, color=PEND,
                         edgecolor=BG, lw=0.4)
        axT.text(start - 0.08, r, "L%d" % ln["i"], fontproperties=mono, color=DIM, ha="right",
                 va="center", clip_on=False)
    for a, b, _ in gaps:
        if b > start and a < end:
            axT.axvspan(a, b, color=CHOP, alpha=0.32, lw=0)
    axT.hlines(len(lines) - 0.5, start, end, color="#2b3550", lw=0.7)
    axT.vlines(onsets[(onsets >= start) & (onsets <= end)], -0.60, -0.25,
               color="#64748b", lw=0.7)
    sung = Rectangle((start, -0.7), 0.0, len(lines) + 0.7, color=SUNGC, alpha=0.20, lw=0,
                     animated=True)
    axT.add_patch(sung)
    active = Rectangle((start, 0.0), 0.0, 0.6, facecolor="none", edgecolor=CUR, lw=1.8,
                       animated=True)
    axT.add_patch(active)
    play = axT.axvline(start, color=LIT, lw=1.2, animated=True)

    # info panel: mix level behind the readouts
    t_all = np.arange(len(rms)) / common.FPS
    m = (t_all >= start) & (t_all <= end)
    axI.fill_between(t_all[m], 0, rms[m], color="#334155", lw=0)
    axI.set_xlim(start, end)
    axI.set_ylim(0, 1.05)
    t_t = axI.text(0.004, 0.92, "", transform=axI.transAxes, fontproperties=mono, color=LIT,
                   va="top", ha="left", animated=True)
    ln_t = axI.text(0.004, 0.06, "", transform=axI.transAxes, fontproperties=mono_jp,
                    color="#a6b0c3", va="bottom", ha="left", animated=True)
    u_t = axI.text(0.42, 0.06, "", transform=axI.transAxes, fontproperties=mono_jp, color=CUR,
                   va="bottom", ha="left", animated=True)
    play2 = axI.axvline(start, color=LIT, lw=1.0, alpha=0.6, animated=True)

    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    dyn = [prev_t, next_t, dim_t, lit_t, wipe, sung, active, play, t_t, ln_t, u_t, play2]
    axw = axL.get_window_extent().width
    ttp = TextToPath()

    def width_pt(s: str) -> float:
        return ttp.get_text_width_height_descent(s, jp, ismath=False)[0] if s else 0.0

    n = int(round((end - start) * fps))
    cmd = [str(FFMPEG), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h), "-r", str(fps),
           "-i", "-", "-ss", "%.3f" % start, "-t", "%.3f" % (end - start), "-i",
           str(common.AUDIO), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(path)]
    row_of = {ln["i"]: r for r, ln in enumerate(lines)}
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ff:
        for i in range(n):
            t = start + i / fps
            ln = next((l for l in lines if l["start"] <= t < l["end"]), None)
            if ln is not None:
                li = lines.index(ln)
                prev_t.set_text(lines[li - 1]["text"] if li else "")
                next_t.set_text(lines[li + 1]["text"] if li + 1 < len(lines) else "")
                text = ln["text"]
                if dim_t.get_text() != text:
                    dim_t.set_text(text)
                    lit_t.set_text(text)
                p = progress(ln["words"], t)
                k, frac = int(p), p - int(p)
                wpt = (width_pt(text[:k]) + frac * width_pt(text[k:k + 1])) if k < len(text) \
                    else width_pt(text)
                cw = max(wpt * dpi / 72.0 / axw, 1e-4)
                clip.set_width(cw)
                wipe.set_xdata([0.06 + cw, 0.06 + cw])
                ln_t.set_text("line %-3d conf %.2f   %s"
                              % (ln["i"], ln["words"][0]["conf"], text[:44]))
            else:
                for a in (prev_t, next_t, dim_t, lit_t):
                    a.set_text("")
                wipe.set_xdata([1.5, 1.5])
                ln_t.set_text("line  --   no line is current here")
            u, j = None, -1
            for jj, x in enumerate(units):
                if x["start"] <= t < x["end"]:
                    u, j = x, jj
                    break
            if u is not None:
                sung.set_width(t - start)
                active.set_xy((u["start"], row_of.get(u["line"], 0) - 0.3))
                active.set_width(max(u["end"] - u["start"], 0.0))
                gb = units[j - 1]["end"] if j and units[j - 1]["line"] == u["line"] else None
                chop = ("   <-- after a %.1f s chop" % (u["start"] - gb)
                        if gb is not None and u["start"] - gb >= CHOP_GAP else "")
                u_t.set_text("syllable %3d/%-3d  「%s」  %.2f-%.2f  (%.2f s)%s"
                             % (j + 1, len(units), u["text"], u["start"], u["end"],
                                u["end"] - u["start"], chop))
                t_t.set_text("t %7.2f s   singing" % t)
            else:
                active.set_width(0.0)
                u_t.set_text("--  no syllable claimed at %.2f s  (silence / chop)" % t)
                t_t.set_text("t %7.2f s   SILENCE / CHOP" % t)
            play.set_xdata([t, t])
            play2.set_xdata([t, t])
            fig.canvas.restore_region(bg)
            for a in dyn:
                a.axes.draw_artist(a)
            fig.canvas.blit(fig.bbox)
            ff.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
            if i % max(1, n // 6) == 0:
                print("  frame %5d/%d  t=%.1f s" % (i, n, t), flush=True)
    plt.close(fig)
    return str(path)


def main(argv: list[str]) -> int:
    def opt(name, dflt, cast):
        return cast(argv[argv.index(name) + 1]) if name in argv else dflt

    start = opt("--start", 60.0, float)
    end = opt("--end", 120.0, float)
    fps = opt("--fps", 20, int)
    src = opt("--src", "stem", str)
    D = load(src, opt("--data", None, str))
    font = pick_font(opt("--font", None, str))
    units = in_window(D["units"], start, end)
    lines = [ln for ln in D["doc"]["lines"] if ln["start"] < end and ln["end"] > start]
    gaps = [g for g in unit_gaps(D["units"]) if g[1] > start and g[0] < end]
    print("  %.1f-%.1f s:  %d syllables, %d lines, %d chops >= %.2f s"
          % (start, end, len(units), len(lines), len(gaps), CHOP_GAP))
    stem = "pilot_%03d_%03d%s" % (round(start), round(end),
                                  "" if src == "stem" else "_" + src)
    print(strip(start, end, D, font, common.OUT / (stem + "_strip.png")))
    if "--no-video" not in argv:
        print(video(start, end, D, font, common.OUT / (stem + ".mp4"), fps=fps))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
