#!/usr/bin/env python3
"""Word-level lyric timing + BPM-split syllable spans -> data/lyrics.approx.json.

The legacy pass (lyric_units.py / align_lyrics.py) infers mora weights from the
orthography and DP-matches them to note onsets; when the guesses miss early the
whole line stretches (tail lines 30-40 s, conf ~0.2). This pass instead leans on
the two sources that already know the answer:

  word level  : whisper (stable-ts + faster-whisper) hears the vocal stem and
                reports word timestamps; the KNOWN text of audio/lyrics.txt is
                matched onto those words character-wise (digitized Japanese/
                Latin both survive NFKC fold). Words are the whitespace tokens
                of a lyric line, so ' '.join(words) == text -- the invariant
                Lyrics.lineCharProgress relies on. Words whisper badly
                mishears land in a plain interpolation between matched
                neighbours (marked by conf=0.5), never in silent intro/outro.
  syllables   : audio/syllable.txt holds the sung syllable sequence of most
                lines (69 of 74; the 5 outro lines get one generated token per
                word). The sequence is grouped onto the display words via
                data/syllable_map.json (an editable draft: move tokens between
                neighbouring words to fix a grouping) and the cut points inside
                a word sit on the BPM lattice (features_*.npz: 128.803 bpm ->
                1/8 note = 0.233 s). Spans are snapped to that lattice, clamped
                monotone inside [word.start, word.end], never crossing a word
                boundary and never below 20 ms.

Engine schema (reference/pdoom-video/app/src/engine/lyrics.ts):
  {"lines": [{"i","text","start","end",
              "words": [{"w","start","end","conf","syl": [[ts,te], ...]}]}],
   "extras": [], "notes": "...", "source": "...", "method": {...}}
'syl' is omitted for single-syllable words (the engine wipes them linearly);
when present it always tiles the word exactly (first start / last end snap to
the word, cut points between).

Modes (first positional arg; default "all"):
    map    rewrite data/syllable_map.json (draft grouping, edit before align)
    align  run/reuse cached word timings; build data/lyrics.approx.json + report
    check  verify an output file against the engine contracts
    all    map (if missing) -> align -> check

Flags: --map PATH --out PATH --model NAME --engine {faster-whisper,openai}
       --device {cuda,cpu} --subdiv N --bpm B --phase P --period S --force

Examples:
    .venv/bin/python analysis/word_syllables.py map
    .venv/bin/python analysis/word_syllables.py align --model large-v3-turbo
    .venv/bin/python analysis/word_syllables.py check
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import common

# stable-ts decodes audio by spawning `ffmpeg`; use the repo's static build
os.environ["PATH"] = str(common.TOOLS / "ff") + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402  (after common + PATH)

PROG = "analysis/word_syllables.py"
SYL_MAP = common.DATA / "syllable_map.json"
DEFAULT_OUT = common.DATA / "lyrics.approx.json"
COMMON_OUT_REL = "data/lyrics.approx.json"
REPORT = common.QA / "word_syllables_report.txt"
FEATURES = common.WORK / "features_vocal.npz"

SUBDIV_DEFAULT = 2.0      # boundary lattice = beat / 2 = 1/8 note (0.233 s)
MIN_SYL_SEC = 0.02        # one syllable span is never below this
TS = 3                    # timestamp decimals (legacy writes 3)

KEEP_RE = re.compile(
    r"[a-z0-9\u00c0-\u024f\u3040-\u30ff\u3005\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
VOWEL_RUN_RE = re.compile(r"[aeiouy]+", re.I)
SMALL_KANA = set("ゃゅょぁぃぅぇぉっゎャュョァィゥェォッヮ")
HAN = re.compile(r"[\u3005\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
HIRA = re.compile(r"[\u3040-\u30fa\u30fc]")
KATA = re.compile(r"[\u30fb-\u30ff]")
LATN = re.compile(r"[A-Za-z0-9\u00c0-\u024f]")


def r3(x: float) -> float:
    return round(float(x), TS)


# --------------------------------------------------------------------------
# orthography helpers (source text is seen raw); fold() is only the matching
# key against the aligned pieces
# --------------------------------------------------------------------------
def fold(s: str) -> str:
    """NFKC + lowercase, punctuation/spacing dropped -- matching key."""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if KEEP_RE.match(ch))


def _runs_by_class(text: str):
    """Group kept chars into runs of K(anji) H(iragana) T(katakana) L(atin)."""
    runs: list[tuple[str, str]] = []
    for ch in unicodedata.normalize("NFKC", text):
        if HAN.match(ch):
            cls = "K"
        elif KATA.match(ch):
            cls = "T"
        elif HIRA.match(ch):
            cls = "H"
        elif LATN.match(ch):
            cls = "L"
        else:
            continue
        if runs and runs[-1][0] == cls:
            runs[-1] = (cls, runs[-1][1] + ch)
        else:
            runs.append((cls, ch))
    return runs
def script_chunks(text: str) -> int:
    """Heuristic count of reading tokens in one display word.

    A chunk is a script run, with a kanji run absorbing the kana run that
    follows (okurigana) -- e.g. '心|に|渦巻く' -> '心に' + '渦巻く'. Good enough
    for a draft; data/syllable_map.json exists precisely so this is revisable.
    """
    runs = _runs_by_class(text)
    if not runs:
        return 0
    n, i = 0, 0
    while i < len(runs):
        n += 1
        if runs[i][0] == "K" and i + 1 < len(runs) and runs[i + 1][0] == "H":
            i += 2
        else:
            i += 1
    return n


def mora_weight(text: str) -> float:
    """Rough 'how much reading does this word carry' weight (morae-ish)."""
    total = 0.0
    for cls, run in _runs_by_class(text):
        if cls == "L":
            total += max(1.0, float(len(VOWEL_RUN_RE.findall(run))))
        elif cls == "K":
            total += 2.0 * len(run)
        else:  # kana: 1 per mora-ish char, small kana ride along
            for j, ch in enumerate(run):
                if ch in SMALL_KANA and j > 0:
                    total += 0.5
                else:
                    total += 1.0
    return total


def alloc_counts(weights: list[float], total: int) -> list[int]:
    """Split `total` tokens over words: >=1 each when possible (0's allowed
    when the sung sequence is shorter than the display words)."""
    n = len(weights)
    if n == 0 or total <= 0:
        return [0] * n
    min_each = 1 if total >= n else 0
    base = [min_each] * n
    rest = total - min_each * n
    if rest == 0:
        return base
    sw = float(sum(weights))
    share = [(w / sw * rest) if sw > 0 else rest / n for w in weights]
    add = [int(math.floor(s)) for s in share]
    rem = rest - sum(add)
    order = sorted(range(n), key=lambda i: share[i] - add[i], reverse=True)
    for i in order[:rem]:
        add[i] += 1
    return [b + a for b, a in zip(base, add)]


# --------------------------------------------------------------------------
# text sources: display words (engine words) and the sung syllable sequence
# --------------------------------------------------------------------------
def display_words(text: str) -> list[str]:
    """Engine words == whitespace tokens of a lyric line."""
    return text.split()


def lyric_lines() -> list[str]:
    lines = [ln.rstrip() for ln in common.LYRICS_TXT.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln.strip()]


def reading_lines() -> dict[int, list[str]]:
    """Syllable.txt tokens per lyric line index (blank lines are separators;
    readings run in order over the first N lyric lines)."""
    raw = common.SYLLABLE_TXT.read_text(encoding="utf-8").splitlines()
    reads: dict[int, list[str]] = {}
    idx = 0
    for ln in raw:
        toks = ln.split()
        if not toks:
            continue
        reads[idx] = toks
        idx += 1
    return reads


def generated_tokens(words: list[str]) -> list[str]:
    """Fallback reading for the outro lines: one token per display word."""
    out = []
    for w in words:
        tok = w.strip("()[]{}…~!?,.:;\"'’‘“”").lower()
        out.append(tok if tok else w.lower())
    return out
# --------------------------------------------------------------------------
# data/syllable_map.json: which reading tokens each display word owns
# --------------------------------------------------------------------------
def build_map(lines: list[str], reads: dict[int, list[str]]) -> dict:
    out_lines = []
    stats = {"chunks": 0, "ratio": 0, "generated": 0, "sparse": 0}
    for i, text in enumerate(lines):
        words = display_words(text)
        generated = i not in reads
        toks = generated_tokens(words) if generated else reads[i]
        n = len(toks)
        chunk_counts = [script_chunks(w) for w in words]
        if (not generated) and sum(chunk_counts) == n:
            counts, method = chunk_counts, "chunks"
        else:
            counts = alloc_counts([mora_weight(w) for w in words], n)
            method = "generated" if generated else "ratio"
        ws, pos = [], 0
        for w, c in zip(words, counts):
            ws.append({"w": w, "syl": toks[pos:pos + c]})
            pos += c
        sparse = any(c == 0 for c in counts)
        stats[method] = stats.get(method, 0) + 1
        if sparse:
            stats["sparse"] += 1
        out_lines.append({
            "i": i, "text": text, "method": method,
            "generated": generated, "sparse": sparse, "words": ws,
        })
    return {
        "source": {"lyrics": "audio/lyrics.txt", "syllables": "audio/syllable.txt"},
        "note": (
            "Draft grouping of the sung syllable sequence (audio/syllable.txt, "
            "'syl' tokens in sung order) onto the whitespace words of "
            "audio/lyrics.txt. To fix a grouping, move tokens between "
            "neighbouring words; the flattened 'syl' list per line must stay "
            "identical to the sung sequence or analysis/word_syllables.py "
            "refuses the file. method=chunks/ratio/generated records how each "
            "line's draft split was guessed. Lines flagged 'generated' have no "
            "sung reading (outro) and use one token per word."
        ),
        "stats": stats,
        "lines": out_lines,
    }


def validate_map(map_doc: dict, lines: list[str], reads: dict[int, list[str]]) -> list[str]:
    errors: list[str] = []
    mlines = map_doc.get("lines", [])
    if len(mlines) != len(lines):
        errors.append(f"map has {len(mlines)} lines, lyrics.txt has {len(lines)}")
        return errors
    for i, text in enumerate(lines):
        entry = mlines[i]
        if entry.get("i") != i:
            errors.append(f"line {i}: map 'i' is {entry.get('i')}")
        if entry.get("text") != text:
            errors.append(f"line {i}: map 'text' != lyrics.txt")
        words = display_words(text)
        got = [w.get("w") for w in entry.get("words", [])]
        if got != words:
            errors.append(f"line {i}: map words {got!r} != lyric words {words!r}")
            continue
        flat = [t for w in entry["words"] for t in w.get("syl", [])]
        want = reads.get(i)
        if want is not None and flat != want:
            errors.append(
                f"line {i}: map 'syl' flattens to {flat!r} but the sung "
                f"sequence is {want!r}"
            )
    return errors


def load_map(lines: list[str], reads: dict[int, list[str]], map_path: Path, rewrite: bool) -> dict:
    if not rewrite and map_path.exists():
        map_doc = json.loads(map_path.read_text(encoding="utf-8"))
        errors = validate_map(map_doc, lines, reads)
        if errors:
            raise SystemExit(
                f"{map_path} is inconsistent with audio/: \n  "
                + "\n  ".join(errors[:12])
                + ("\n  ..." if len(errors) > 12 else "")
                + "\nregenerate with: analysis/word_syllables.py map"
            )
        return map_doc
    map_doc = build_map(lines, reads)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(
        json.dumps(map_doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return map_doc
# --------------------------------------------------------------------------
# word-level alignment: stable-ts forced alignment of the known lyric text
# against audio/vocals.mp3, then matched back to the display words char-wise
# --------------------------------------------------------------------------
def word_cache_path(model_name: str) -> Path:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name)
    return common.WORK / f"word_times_{tag}.json"


def run_align(model_name: str, engine: str, device: str, text: str,
              locate: str = "transcribe") -> tuple[list, dict]:
    """Forced-align `text` to the vocal stem. Returns (pieces, meta); a piece
    is [word_text, start, end, probability].

    locate="transcribe": whisper hears the song and reports word timestamps;
        the known lyric text is then matched onto those words char-wise (the
        chosen path -- whole-file `align` scatters text that whisper mishears
        into the silent intro).
    locate="align": stable-ts forced alignment of the text over the whole file
        (kept as an option; tends to glue failed phrases to noise).
    """
    import stable_whisper

    t0 = time.time()
    if engine == "faster-whisper":
        model = stable_whisper.load_faster_whisper(
            model_name, device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
    elif engine == "openai":
        model = stable_whisper.load_model(model_name, device=device)
    else:  # auto -> faster-whisper backend when installed
        try:
            model = stable_whisper.load_faster_whisper(
                model_name, device=device,
                compute_type="float16" if device == "cuda" else "int8",
            )
            engine = "faster-whisper"
        except Exception:
            model = stable_whisper.load_model(model_name, device=device)
            engine = "openai"
    if locate == "transcribe":
        result = model.transcribe(str(common.VOCALS), language="ja",
                                  word_timestamps=True, verbose=False, vad=True)
    else:
        result = model.align(str(common.VOCALS), text, language="ja", verbose=False)
    if result is None:
        raise SystemExit(f"stable-ts {locate} failed (returned None)")
    pieces = []
    for seg in result.segments:
        for w in (seg.words or []):
            pieces.append([str(w.word), float(w.start), float(w.end),
                           float(w.probability if w.probability is not None else 0.0)])
    meta = {
        "aligner": f"stable-ts {getattr(stable_whisper, '__version__', '?')}",
        "engine": engine,
        "model": model_name,
        "device": device,
        "language": "ja",
        "locate": locate,
        "n_pieces": len(pieces),
        "seconds": round(time.time() - t0, 1),
    }
    return pieces, meta


def align_words(lines: list[str], model_name: str, engine: str, device: str,
                cache: Path, force: bool, locate: str = "transcribe") -> tuple[list, dict]:
    """Return (pieces, meta); pieces are cached under analysis/work."""
    text = "\n".join(" ".join(display_words(ln)) for ln in lines)
    if cache.exists() and not force:
        doc = json.loads(cache.read_text(encoding="utf-8"))
        meta = doc.get("meta", {})
        if meta.get("text") == text and meta.get("locate", "align") == locate:
            return doc["pieces"], meta
        print(f"  cache {cache.name} is stale (source text/locate changed) -> re-running")
    pieces, meta = run_align(model_name, engine, device, text, locate=locate)
    meta["text"] = text
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"meta": meta, "pieces": pieces}, ensure_ascii=False),
                     encoding="utf-8")
    return pieces, meta
def link_words(lines: list[str], pieces: list) -> tuple[list, dict]:
    """Per line, per word [start, end, prob] via normalized char matching.

    whisper splits/mishears words, so the link runs on the folded character
    streams; a word's span is the BIGGEST TIME CLUSTER of its matched pieces
    (stray matches -- kana collide across unrelated text -- are ignored).
    Junk pieces (degenerate durations) are dropped first."""

    good = []
    for p in pieces:
        dur = float(p[2]) - float(p[1])
        if dur <= 0.0:
            continue
        if dur > 2.5 and float(p[3]) < 0.3:      # hallucinated blob
            continue
        good.append(p)
    src_chars: list[tuple[int, int]] = []   # char -> (line, word)
    for li, ln in enumerate(lines):
        for wi, w in enumerate(display_words(ln)):
            for _ in fold(w):
                src_chars.append((li, wi))
    src_stream = "".join(
        ch for ln in lines for w in display_words(ln) for ch in fold(w)
    )
    ali_chars: list[int] = []               # char -> piece idx
    ali_stream = []
    for pi, (txt, _s, _e, _p) in enumerate(good):
        for ch in fold(txt):
            ali_chars.append(pi)
            ali_stream.append(ch)
    ali_stream_s = "".join(ali_stream)

    acc: dict[tuple[int, int], list] = {}   # (li,wi) -> [start, end, probsum, hits]
    matched = 0
    matcher = difflib.SequenceMatcher(None, src_stream, ali_stream_s, autojunk=False)
    for blk in matcher.get_matching_blocks():
        for k in range(blk.size):
            li, wi = src_chars[blk.a + k]
            pi = ali_chars[blk.b + k]
            _txt, st, en, pr = good[pi]
            acc.setdefault((li, wi), []).append((st, en, pr))
            matched += 1

    out: list[list] = []
    for li, ln in enumerate(lines):
        row = []
        for wi, w in enumerate(display_words(ln)):
            hits = acc.get((li, wi))
            if not hits:
                row.append(None)
                continue
            hits.sort()
            dl = max(1, len(fold(w)))
            clusters: list[list] = []       # [start, end, chars, probsum]
            for st, en, pr in hits:
                if clusters and st - clusters[-1][1] <= 1.2:
                    c = clusters[-1]
                    c[1] = max(c[1], en)
                    c[2] += 1
                    c[3] += pr
                else:
                    clusters.append([st, en, 1, pr])

            def plausible(c: list) -> bool:
                chars = c[2]
                dur = c[1] - c[0]
                pr = c[3] / chars
                if dur < 0.06 * chars:          # absurdly fast -> junk
                    return False
                if dur > 1.5 + 0.8 * chars:     # absurdly stretched -> junk
                    return False
                if pr < 0.05:                   # whisper didn't hear it
                    return False
                if chars < 0.34 * dl and pr < 0.25:  # stray kana collision
                    return False
                return True

            best = None
            for c in sorted(clusters, key=lambda c: c[2], reverse=True):
                if plausible(c):
                    best = c
                    break
            if best is None:
                row.append(None)
            else:
                row.append([r3(best[0]), r3(best[1]), round(best[3] / best[2], 3)])
        out.append(row)
    link_meta = {
        "src_chars": len(src_stream),
        "ali_chars": len(ali_stream_s),
        "matched_chars": matched,
        "match_ratio": round(matched / max(1, len(src_stream)), 4),
    }
    return out, link_meta


def interpolate_missing(times: list[list], lines: list[str]) -> list[list]:
    """Fill unlinked words from their linked neighbours inside each line; a
    run of fully unlinked lines is sprayed over the vocal-activity runs inside
    the gap between the surrounding linked lines."""
    n_lines = len(times)
    for li, row in enumerate(times):
        known = [(wi, t) for wi, t in enumerate(row) if t is not None]
        if not known:
            continue
        for wi, t in enumerate(row):
            if t is not None:
                continue
            before = [(w2, t2) for w2, t2 in known if w2 < wi]
            after = [(w2, t2) for w2, t2 in known if w2 > wi]
            if before and after:
                lo = before[-1][1][1]
                hi = after[0][1][0]
                span = max(hi - lo, 0.0)
                bucket = after[0][0] - before[-1][0]
                st = lo + span * (wi - before[-1][0]) / max(1, bucket)
                en = lo + span * (wi + 1 - before[-1][0]) / max(1, bucket)
                cap = 1.0 + 0.45 * max(1, len(fold(lines[li].split()[wi])))
                en = min(en, st + cap)
                times[li][wi] = [r3(st), r3(en), 0.0]
            elif before:  # trailing words after the last linked one
                st = before[-1][1][1]
                times[li][wi] = [r3(st), r3(st + 0.45), 0.0]
            else:          # leading words before the first linked one
                hi = after[0][1][0]
                times[li][wi] = [r3(max(0.0, hi - 0.45)), r3(hi), 0.0]
    # fully unlinked line groups -> spray over the vocal-activity runs inside
    # the gap between the surrounding linked lines (never over known silence)
    runs = vocal_runs()
    i = 0
    while i < n_lines:
        if all(t is not None for t in times[i]):
            i += 1
            continue
        g = i
        while g < n_lines and all(t is None for t in times[g]):
            g += 1
        lo = 0.0
        for k in range(i - 1, -1, -1):
            flat = [t for t in times[k] if t is not None]
            if flat:
                lo = max(t[1] for t in flat)
                break
        hi = None
        for k in range(g, n_lines):
            flat = [t for t in times[k] if t is not None]
            if flat:
                hi = min(t[0] for t in flat)
                break
        words = [(r, wi) for r in range(i, g) for wi in range(len(times[r]))]
        if hi is None or hi <= lo:
            hi = lo + 3.0 * len(words)
        segs = []
        for a, b in runs:
            aa, bb = max(a, lo), min(b, hi)
            if bb > aa:
                segs.append([aa, bb])
        if not segs:
            segs = [[lo, hi]]
        total = sum(b - a for a, b in segs)
        wl = [(max(1, len(fold(lines[r].split()[wi]))), r, wi) for r, wi in words]
        wsum = float(sum(d for d, _r, _wi in wl))

        def t_at(u: float) -> float:
            u = min(max(u, 0.0), total)
            acc = 0.0
            for a, b in segs:
                if u <= acc + (b - a):
                    return a + (u - acc)
                acc += b - a
            return segs[-1][1]

        u0 = 0.0
        for dl, r, wi in wl:
            u1 = u0 + total * dl / wsum
            st, en = t_at(u0), t_at(u1)
            if en - st > (u1 - u0) + 0.35:
                # one word must never straddle a silence gap -> clamp into the
                # vocal run that contains its start
                for a, b in segs:
                    if a - 0.001 <= st <= b + 0.001:
                        en = min(en, b)
                        break
            en = min(en, st + 1.0 + 0.45 * dl)   # plausible singing width
            if en <= st:
                en = st + MIN_SYL_SEC
            times[r][wi] = [r3(st), r3(en), 0.0]
            u0 = u1
    return times


def refine_lines(model_name: str, engine: str, device: str, lines: list[str],
                 times: list[list]) -> tuple[list[list], int]:
    """Second pass for implausible lines (all-interpolated, one word eating the
    line, or a line wider than 6.5 s): fit JUST that line's text inside a
    window around the first placement. The window is bounded by the
    neighbouring lines so refinement cannot invert order."""
    import stable_whisper
    from stable_whisper.audio import SAMPLE_RATE, load_audio

    bad = []
    for li, row in enumerate(times):
        durs = [t[1] - t[0] for t in row]
        if max(durs) > 6.5 or all(t[2] <= 0.0 for t in row) or (
                row[-1][1] - row[0][0]) > 12.0 or (
                len(row) > 1 and max(durs) > 0.75 * max(0.01, sum(durs))):
            bad.append(li)
    if not bad:
        return times, 0

    if engine == "faster-whisper":
        model = stable_whisper.load_faster_whisper(
            model_name, device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
    else:
        model = stable_whisper.load_model(model_name, device=device)
    audio = load_audio(str(common.VOCALS), sr=SAMPLE_RATE, mono=True)
    dur = len(audio) / SAMPLE_RATE

    n_refined = 0
    for li in bad:
        row = times[li]
        floor = times[li - 1][-1][1] if li > 0 else 0.0
        ceil = times[li + 1][0][0] if li + 1 < len(times) else dur
        lo = max(floor - 2.0, row[0][0] - 25.0, 0.0)
        hi = min(ceil + 2.0, row[-1][1] + 25.0, dur)
        if hi - lo < 1.0:
            continue
        seg = np.asarray(audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)],
                         dtype=np.float32)
        text = " ".join(display_words(lines[li]))
        try:
            res = model.align(seg, text, language="ja", verbose=False)
        except Exception:  # noqa: BLE001
            continue
        if res is None:
            continue
        pieces = []
        for sg in res.segments:
            for w in (sg.words or []):
                pieces.append([str(w.word), float(w.start), float(w.end),
                               float(w.probability if w.probability is not None else 0.0)])
        if not pieces:
            continue
        sub, _ = link_words([lines[li]], pieces)
        if any(t is None for t in sub[0]):
            continue
        cand = [[r3(t[0] + lo), r3(t[1] + lo), t[2]] for t in sub[0]]
        if max(e - s for s, e, _ in cand) > 12.0:
            continue
        times[li] = cand
        n_refined += 1
    return times, n_refined


# --------------------------------------------------------------------------
# BPM lattice from analysis/work/features_*.npz (bpm=128.803, phase=0.410)
# --------------------------------------------------------------------------
def load_grid(bpm: float | None, phase: float | None, period: float | None) -> dict:
    z = np.load(FEATURES)
    g = {
        "bpm": float(z["bpm"]),
        "period": float(z["period"]),
        "phase": float(z["phase"]),
        "source": "analysis/work/features_vocal.npz",
    }
    if bpm is not None:
        g["bpm"] = float(bpm)
        g["period"] = 60.0 / float(bpm)
    if period is not None:
        g["period"] = float(period)
        g["bpm"] = 60.0 / float(period)
    if phase is not None:
        g["phase"] = float(phase)
    return g


def vocal_runs() -> list[tuple[float, float]]:
    z = np.load(FEATURES)
    runs = z["runs"]
    return [(float(a), float(b)) for a, b in runs[:, :2]]


def run_coverage(s: float, e: float, runs: list[tuple[float, float]]) -> float:
    if e <= s:
        return 0.0
    hit = 0.0
    for a, b in runs:
        hit += max(0.0, min(e, b) - max(s, a))
    return min(1.0, hit / (e - s))


def round_grid(t: float, phase: float, step: float) -> float:
    return phase + round((t - phase) / step) * step


def split_on_grid(s: float, e: float, n: int, phase: float, period: float,
                  subdiv: float) -> tuple[list[list[float]], float]:
    """Cut [s, e] into n spans (n-1 interior boundaries). Boundaries start at
    the uniform ideal positions and are snapped to the BPM lattice
    (phase + k*period/subdiv), then clamped strictly monotone inside the word
    with >= MIN_SYL_SEC per span. Returns (spans, mean |snap - ideal|)."""
    s, e = float(s), float(e)
    if n <= 1:
        return [[s, e]], 0.0
    if e <= s:
        e = s + MIN_SYL_SEC
    ideal = [s + (e - s) * k / n for k in range(1, n)]
    step = period / float(subdiv)
    fixed: list[float] = []
    residual = 0.0
    prev = s
    for j, t in enumerate(ideal):
        snapped = round_grid(t, phase, step)
        residual += abs(snapped - t)
        hi = (ideal[j + 1] if j + 1 < len(ideal) else e) - MIN_SYL_SEC
        lo = prev + MIN_SYL_SEC
        if hi <= lo:            # degenerate word (too short for n tokens)
            t = prev + MIN_SYL_SEC
        else:
            t = min(max(snapped, lo), hi)
        fixed.append(t)
        prev = t
    cuts = [s] + fixed + [e]
    spans = [[cuts[k], cuts[k + 1]] for k in range(n)]
    if any(b - a <= 0.0 for a, b in spans):  # last resort: plain uniform
        spans = [[s + (e - s) * k / n, s + (e - s) * (k + 1) / n] for k in range(n)]
    return [[r3(a), r3(b)] for a, b in spans], residual / max(1, n - 1)
# --------------------------------------------------------------------------
# engine-schema output
# --------------------------------------------------------------------------
def emit_word(w: str, ts: float, te: float, toks: list[str], grid: dict, subdiv: float,
              runs: list, prob: float) -> tuple[dict, float]:
    """One engine Word; 'syl' is tiled inside [ts, te] and omits single spans."""  # noqa
    ts, te = float(ts), float(te)
    if te < ts + MIN_SYL_SEC:
        te = ts + MIN_SYL_SEC
    n = len(toks)
    entry = {"w": w, "start": r3(ts), "end": r3(te)}
    res = 0.0
    if n >= 1:
        spans, res = split_on_grid(ts, te, n, grid["phase"], grid["period"], subdiv)
        if n > 1:
            entry["syl"] = spans
        # single-syllable words: no 'syl' (engine wipes them linearly)
    cov = run_coverage(ts, te, runs)
    base = prob if prob > 0 else 0.5
    entry["conf"] = round(max(0.0, min(1.0, base * (0.5 + 0.5 * cov))), 3)
    return entry, res


def shift_line(line: dict, d: float) -> None:
    """Move a whole line (words + syllable spans) right by d seconds."""
    line["start"] = r3(line["start"] + d)
    line["end"] = r3(line["end"] + d)
    for w in line["words"]:
        w["start"] = r3(w["start"] + d)
        w["end"] = r3(w["end"] + d)
        for s in w.get("syl", []):
            s[0] = r3(s[0] + d)
            s[1] = r3(s[1] + d)


def build_doc(lines: list[str], mapping: dict, times: list[list], grid: dict,
              subdiv: float, runs: list, method: dict) -> tuple[dict, dict]:
    out_lines = []
    line_probs: list = []
    res_sum = res_n = 0
    n_syl_words = 0
    for i, text in enumerate(lines):
        words = display_words(text)
        mwords = mapping["lines"][i]["words"]
        w_out: list[dict] = []
        raw: list[list[float]] = []
        probs: list[float] = []
        for w, mw, (ts, te, pr) in zip(words, mwords, times[i]):
            ts, te = float(ts), float(te)
            if te < ts + MIN_SYL_SEC:
                te = ts + MIN_SYL_SEC
            if raw:
                if ts < raw[-1][0]:            # out of order -> push right
                    ts = raw[-1][0]
                if ts < raw[-1][1]:            # overlap: close prev, else push this
                    if ts - raw[-1][0] >= MIN_SYL_SEC:
                        raw[-1][1] = ts
                    else:
                        ts = raw[-1][1]
                        te = max(te, ts + MIN_SYL_SEC)
            raw.append([ts, te])
            probs.append(float(pr or 0.0))
        line_probs.append(probs)
        for (w, mw, (ts, te), pr) in zip(words, mwords, raw, probs):
            entry, res = emit_word(w, ts, te, mw["syl"], grid, subdiv, runs, pr)
            res_sum += res
            res_n += 1
            if len(mw["syl"]) > 1:
                n_syl_words += 1
            w_out.append(entry)
        out_lines.append({
            "i": i,
            "text": text,
            "start": w_out[0]["start"],
            "end": w_out[-1]["end"],
            "words": w_out,
        })
    # lines must not overlap (engine lineAt returns the first match) -> clamp
    for i in range(len(out_lines) - 1):
        nxt = out_lines[i + 1]["start"]
        if out_lines[i]["end"] > nxt:
            prev = out_lines[i]
            wlast = prev["words"][-1]
            if nxt > wlast["start"]:
                toks = mapping["lines"][i]["words"][-1]["syl"]
                w2, _ = emit_word(wlast["w"], wlast["start"], nxt, toks,
                                  grid, subdiv, runs, line_probs[i][-1])
                prev["words"][-1] = w2
                prev["end"] = w2["end"]
            # else: pathological (next line starts before this word does) --
            # handled by the shift pass below
    # min-duration floors can still leave tiny overlaps -> shift later lines
    # right (cascade forward; gaps in real singing absorb the shift)
    for i in range(len(out_lines) - 1):
        prev, nxt = out_lines[i], out_lines[i + 1]
        if prev["end"] > nxt["start"]:
            shift_line(nxt, round(prev["end"] - nxt["start"] + 0.001, 3))
    doc = {
        "lines": out_lines,
        "extras": [],
        "notes": (
            "Word-level timestamps: stable-ts forced alignment of the known "
            "text of audio/lyrics.txt against audio/vocals.mp3 (see method). "
            "Syllables: the sung sequence of audio/syllable.txt grouped onto "
            "the display words via data/syllable_map.json; cut points inside a "
            "word are snapped to the 1/8-note BPM lattice, clamped monotone, "
            ">= 20 ms, and never cross a word boundary. 'syl' is omitted for "
            "single-syllable words (engine wipes them linearly) and otherwise "
            "tiles the word exactly. conf = alignment confidence x "
            "in-vocal-coverage factor (0.5+0.5*cov). Line order IS performance "
            "order."
        ),
        "source": PROG,
        "method": method,
    }
    stats = {
        "lines": len(out_lines),
        "words": sum(len(l["words"]) for l in out_lines),
        "syl_words": n_syl_words,
        "snap_residual_mean": round(res_sum / max(1, res_n), 4),
        "span": [out_lines[0]["start"], out_lines[-1]["end"]],
    }
    return doc, stats
# --------------------------------------------------------------------------
# QA report
# --------------------------------------------------------------------------
def write_report(path: Path, doc: dict, stats: dict, method: dict, grid: dict,
                 map_doc: dict, align_meta: dict, link_meta: dict,
                 interp_words: int, anomalies: list[str]) -> None:
    L = doc["lines"]
    confs = [w["conf"] for ln in L for w in ln["words"]]
    low_conf = sum(1 for c in confs if c < 0.5)
    ms = map_doc["stats"]
    rows = []
    for ln, mL in zip(L, map_doc["lines"]):
        counts = [len(w["syl"]) for w in mL["words"]]
        conf = round(sum(w["conf"] for w in ln["words"]) / max(1, len(ln["words"])), 3)
        tok = "|".join(str(c) for c in counts)
        rows.append(
            f"  {ln['i']:>2}  {ln['start']:>7.3f} {ln['end']:>7.3f} "
            f"{ln['end'] - ln['start']:>6.2f} {len(ln['words']):>3} "
            f"{sum(counts):>3}  {mL['method']:<9} {conf:>5.2f}  {tok}"
        )
    review = [
        f"  {mL['i']:>2}  {mL['text'][:52]:<52} words={len(mL['words'])} "
        f"tokens={sum(len(w['syl']) for w in mL['words'])} "
        f"-> {'|'.join(str(len(w['syl'])) for w in mL['words'])}"
        for mL in map_doc["lines"]
        if mL["method"] != "chunks" and not mL["generated"]
    ]
    gen = [
        f"  {mL['i']:>2}  {mL['text'][:52]}"
        for mL in map_doc["lines"] if mL["generated"]
    ]
    text = "\n".join([
        "word_syllables report",
        "=====================",
        f"generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"outputs   : {COMMON_OUT_REL} + analysis/qa/word_syllables_report.txt",
        "sources   : audio/lyrics.txt (display words) x audio/vocals.mp3 (timing)",
        "            audio/syllable.txt (sung syllable sequence) + data/syllable_map.json (grouping)",
        f"map       : chunks={ms.get('chunks', 0)} ratio={ms.get('ratio', 0)} "
        f"generated={ms.get('generated', 0)} sparse={ms.get('sparse', 0)} "
        f"(edit data/syllable_map.json to fix a grouping)",
        f"align     : {align_meta.get('aligner')} | {align_meta.get('model')} | "
        f"{align_meta.get('engine')} | {align_meta.get('device')} | "
        f"locate={align_meta.get('locate', '?')} | "
        f"{align_meta.get('n_pieces')} pieces in {align_meta.get('seconds')} s",
        f"            char match {link_meta.get('match_ratio', 0):.1%} "
        f"({link_meta.get('matched_chars', 0)}/{link_meta.get('src_chars', 0)} chars); "
        f"words interpolated elsewhere: {interp_words}",
        f"grid      : {grid['bpm']:.3f} bpm, period {grid['period']:.4f} s, "
        f"phase {grid['phase']:.3f} s  ({grid['source']})",
        f"            split lattice = period/{method['subdiv']:g} = "
        f"{grid['period'] / float(method['subdiv']):.4f} s; mono-clamp, min {MIN_SYL_SEC * 1000:.0f} ms",
        f"span      : lines {stats['span'][0]} - {stats['span'][1]} s "
        f"({stats['lines']} lines, {stats['words']} words, {stats['syl_words']} multi-syllable)",
        f"snap      : mean |snap - ideal| = {stats['snap_residual_mean']} s",
        f"conf      : mean {sum(confs) / max(1, len(confs)):.3f}, min {min(confs):.3f} "
        f"(words below 0.5: {low_conf})",
        f"monotone  : word/line spans clamped non-overlapping inside the line and across lines",
        "",
        "lines",
        "  i    start     end    dur  wd tok method    conf  tokens-per-word",
        *rows,
        "",
        "review (grouping guessed proportionally; some of these may need a token moved in the map)",
        *(review or ["  (none)"]),
        "",
        "generated readings (outro lines have no audio/syllable.txt entry)",
        *(gen or ["  (none)"]),
        "",
        "anomalies",
        *(anomalies or ["  (none)"]),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
# --------------------------------------------------------------------------
# engine contracts
# --------------------------------------------------------------------------
def check_doc(doc_path: Path, map_path: Path) -> int:
    fails: list[str] = []
    oks = 0

    def need(cond: bool, label: str) -> None:
        nonlocal oks
        if cond:
            oks += 1
        else:
            fails.append(label)

    try:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  cannot parse {doc_path}: {exc}")
        return 1
    lines = lyric_lines()
    reads = reading_lines()
    L = doc.get("lines", [])
    need(len(L) == len(lines) == 74, f"74 lines (got {len(L)} / lyrics {len(lines)})")
    need("extras" in doc and "notes" in doc and "source" in doc, "notes/source/extras present")

    map_doc = None
    if map_path.exists():
        map_doc = json.loads(map_path.read_text(encoding="utf-8"))
        for err in validate_map(map_doc, lines, reads)[:5]:
            fails.append(f"map: {err}")

    prev_line_end = -1.0
    for i, ln in enumerate(L):
        text = lines[i] if i < len(lines) else None
        need(ln.get("i") == i, f"line {i}: sequential i")
        need(ln.get("text") == text, f"line {i}: text == lyrics.txt")
        words = ln.get("words", [])
        need(" ".join(w.get("w", "") for w in words) == ln.get("text"),
             f"line {i}: ' '.join(words) == text")
        need(words and ln.get("start") == words[0].get("start")
             and ln.get("end") == words[-1].get("end"),
             f"line {i}: start/end == first/last word")
        need(ln.get("start", 0.0) >= prev_line_end - 1e-9,
             f"line {i}: starts at/after previous line end ({prev_line_end})")
        prev_line_end = float(ln.get("end", 0.0))
        prev_end = -1.0
        for w in words:
            s, e = float(w.get("start", -1.0)), float(w.get("end", -1.0))
            need(s >= 0.0 and e > s, f"line {i} '{w.get('w')}': start < end")
            need(s >= prev_end - 1e-9,
                 f"line {i} '{w.get('w')}': non-overlapping words (prev end {prev_end})")
            need("conf" in w and 0.0 <= float(w["conf"]) <= 1.0,
                 f"line {i} '{w.get('w')}': conf in [0,1]")
            prev_end = e
            syl = w.get("syl")
            tok_n = None
            if map_doc is not None and i < len(map_doc["lines"]):
                mwords = map_doc["lines"][i]["words"]
                for mw, w2 in zip(mwords, words):
                    if w2 is w:
                        tok_n = len(mw.get("syl", []))
            if syl is not None:
                need(len(syl) > 1, f"line {i} '{w.get('w')}': syl only when >1 token")
                if tok_n is not None:
                    need(len(syl) == tok_n,
                         f"line {i} '{w.get('w')}': {len(syl)} syl spans vs {tok_n} tokens")
                need(abs(float(syl[0][0]) - s) <= 5e-4 and abs(float(syl[-1][1]) - e) <= 5e-4,
                     f"line {i} '{w.get('w')}': syl tiles the word (first/last snap)")
                ok_span = True
                for (a, b) in syl:
                    if not (float(b) > float(a)):
                        ok_span = False
                need(ok_span, f"line {i} '{w.get('w')}': positive syllable spans")
                contig = all(
                    abs(float(syl[k][1]) - float(syl[k + 1][0])) <= 5e-4
                    for k in range(len(syl) - 1)
                )
                need(contig, f"line {i} '{w.get('w')}': contiguous syllable tiling")
            else:
                if tok_n is not None:
                    need(tok_n <= 1,
                         f"line {i} '{w.get('w')}': no syl but {tok_n} tokens")

    print(f"check {doc_path}: {oks} invariants passed, {len(fails)} failed")
    for f in fails[:20]:
        print("  FAIL " + f)
    if len(fails) > 20:
        print(f"  ... {len(fails) - 20} more")
    return 1 if fails else 0
# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _device(arg: str | None) -> str:
    if arg:
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="word-level lyric timing + BPM-split syllables -> "
                    "data/lyrics.approx.json (engine schema)",
    )
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["all", "map", "align", "check"],
                    help="map=rewrite data/syllable_map.json | align=run timing + "
                         "write output | check=invariants | all=map-if-missing+align+check")
    ap.add_argument("--map", dest="map_path", default=str(SYL_MAP),
                    help="syllable->word grouping file (default data/syllable_map.json)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output json path")
    ap.add_argument("--model", default="large-v3-turbo", help="whisper model name")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "faster-whisper", "openai"])
    ap.add_argument("--locate", default="transcribe", choices=["transcribe", "align"],
                    help="transcribe=whisper word timestamps + char match (default, "
                         "robust); align=whole-file forced alignment")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--subdiv", type=float, default=SUBDIV_DEFAULT,
                    help="syllable lattice = beat/subdiv (2 = 1/8 note)")
    ap.add_argument("--bpm", type=float, default=None, help="override BPM")
    ap.add_argument("--phase", type=float, default=None, help="override beat phase (s)")
    ap.add_argument("--period", type=float, default=None, help="override beat period (s)")
    ap.add_argument("--force", action="store_true", help="ignore the alignment cache")
    args = ap.parse_args(argv)

    map_path = Path(args.map_path)
    out_path = Path(args.out)
    lines = lyric_lines()
    reads = reading_lines()
    device = _device(args.device)

    if args.mode in ("map", "all"):
        rewrite = args.mode == "map"
        if rewrite or not map_path.exists():
            map_doc = load_map(lines, reads, map_path, rewrite=True)
            ms = map_doc["stats"]
            print(f"  wrote {map_path}: chunks={ms['chunks']} ratio={ms['ratio']} "
                  f"generated={ms['generated']} sparse={ms['sparse']}")
            for mL in map_doc["lines"]:
                if mL["method"] != "chunks" and not mL["generated"]:
                    counts = "|".join(str(len(w["syl"])) for w in mL["words"])
                    print(f"    review line {mL['i']:>2}: {mL['text'][:48]!r} -> {counts}")
        if args.mode == "map":
            return 0

    if args.mode in ("align", "all"):
        map_doc = load_map(lines, reads, map_path, rewrite=False)
        cache = word_cache_path(args.model)
        print(f"  aligning {len(lines)} lines to {common.VOCALS.name} "
              f"(model={args.model}, engine={args.engine}, device={device}) ...")
        pieces, align_meta = align_words(lines, args.model, args.engine, device,
                                         cache, args.force, locate=args.locate)
        times, link_meta = link_words(lines, pieces)
        n_interp = sum(1 for row in times for t in row if t is None)
        times = interpolate_missing(times, lines)
        times, n_refined = refine_lines(args.model, args.engine, device, lines, times)
        grid = load_grid(args.bpm, args.phase, args.period)
        runs = vocal_runs()
        method = {
            "aligner": align_meta.get("aligner"),
            "engine": align_meta.get("engine"),
            "locate": align_meta.get("locate"),
            "model": align_meta.get("model"),
            "device": align_meta.get("device"),
            "language": align_meta.get("language"),
            "audio": "audio/vocals.mp3",
            "char_match": link_meta.get("match_ratio"),
            "bpm": round(grid["bpm"], 3),
            "period": round(grid["period"], 4),
            "phase": round(grid["phase"], 3),
            "subdiv": args.subdiv,
            "grid_step": round(grid["period"] / float(args.subdiv), 4),
            "min_syl_sec": MIN_SYL_SEC,
            "refined_lines": n_refined,
            "syllable_map": str(map_path.relative_to(common.ROOT)),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        doc, stats = build_doc(lines, map_doc, times, grid, args.subdiv, runs, method)
        anomalies = []
        for ln in doc["lines"]:
            for w in ln["words"]:
                dur = w["end"] - w["start"]
                if w["conf"] < 0.35:
                    anomalies.append(f"  line {ln['i']:>2} '{w['w']}' conf={w['conf']}")
                if dur > 6.0:
                    anomalies.append(f"  line {ln['i']:>2} '{w['w']}' span {w['start']}-"
                                     f"{w['end']} ({dur:.2f}s)")
                if w["end"] > 601.0:
                    anomalies.append(f"  line {ln['i']:>2} '{w['w']}' ends past audio end")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
        write_report(REPORT, doc, stats, method, grid, map_doc, align_meta,
                     link_meta, n_interp, anomalies)
        print(f"  wrote {out_path} ({stats['lines']} lines, {stats['words']} words, "
              f"{stats['syl_words']} multi-syllable) + {REPORT.relative_to(common.ROOT)}")
        print(f"  link: {link_meta['match_ratio']:.1%} char match, "
              f"{n_interp} words interpolated; grid snap mean "
              f"{stats['snap_residual_mean']}s")

    if args.mode in ("check", "all"):
        return check_doc(out_path, map_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())