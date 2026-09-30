#!/usr/bin/env python3
"""Script-native lyric units: orthography -> mora / syllable weights.

No romanization: Chinese stays Chinese, kanji stays kanji.  A unit's weight is how
many notes (morae / syllables) it can occupy, derived from the script alone:

  kana      one mora per kana glyph.  Small ゃゅょャュョ after a kana = +0 (digraph).
            Small ぁぃぅぇぉァィゥェォ = +1 (long vowel) unless the pair is one of the
            katakana digraphs (ファ ティ シェ ツァ ウィ ヴェ ドゥ ...) = +0.
            っッ (sokuon), ー (chōon), んン (moraic n) = 1 each, own unit.
  CJK       Japanese kanji: 1..3 morae, prior 2 -- the vocal onsets decide the exact
            number (align_lyrics), because in sung Japanese one mora is one note.
            Chinese hanzi: exactly 1.  Mandarin is one syllable per character, so the
            Chinese lines need no reading, no dictionary and no romanization at all.
  々        iteration mark: inherits the previous glyph's weight and joins its unit.
  latin     vowel-group syllable count (heuristic; the onsets refine it later).
  punct     quotes, brackets, ！？… = 0 morae, folded into the neighbouring glyph unit
            so a per-glyph wipe never stalls on a punctuation cell.

A *word* is a whitespace token of the line, verbatim, so ' '.join(words) == text --
the invariant Lyrics.lineCharProgress (app/src/engine/lyrics.ts) relies on.  A *unit*
is one sung glyph group inside a word: the list of units is exactly Word.syl, so
- Lyrics.wordProgress paces the word's wipe by mora (a 2-mora kanji holds twice as
  long as a 1-mora kana), and
- Lyrics.lineCharProgress drives the per-glyph wipe across the whole line.

Language is decided per *run*, never per line and never per character: a CJK run in a
token that contains kana is Japanese, unless the run is listed in zh_runs; a token
with no kana at all is decided by ja_tokens.  Both lists are stored in
data/lyrics.src.json as well, so the judgement is explicit and editable.  A CJK-only
token in neither list is REPORTED, never silently guessed.

Run:  python3 analysis/lyric_units.py            # table + data/lyrics.src.json
      python3 analysis/lyric_units.py table      # unit table only (stdout)
      python3 analysis/lyric_units.py src        # data/lyrics.src.json only
      python3 analysis/lyric_units.py check      # invariants only, non-zero on failure
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))  # legacy/: shared lib lives in analysis/

import common

# ---------------------------------------------------------------- script classes
LONG = "\u30fc"          # ー  chōonpu
ITER = "\u3005"          # 々  iteration mark
DIG0 = set("\u3083\u3085\u3087\u30e3\u30e5\u30e7")                    # ゃゅょャュョ
SMLV = set("\u3041\u3043\u3045\u3047\u3049\u30a1\u30a3\u30a5\u30a7\u30a9")  # ぁぃぅぇぉァィゥェォ
# katakana pairs where the small vowel is a digraph, not a long vowel (0 extra morae)
KATA_DIGRAPH = {
    ("\u30d5", "\u30a1"), ("\u30d5", "\u30a3"), ("\u30d5", "\u30a7"), ("\u30d5", "\u30a9"),   # ファ フィ フェ フォ
    ("\u30f4", "\u30a1"), ("\u30f4", "\u30a3"), ("\u30f4", "\u30a7"), ("\u30f4", "\u30a9"),   # ヴァ ヴィ ヴェ ヴォ
    ("\u30a6", "\u30a3"), ("\u30a6", "\u30a7"), ("\u30a6", "\u30a9"),                         # ウィ ウェ ウォ
    ("\u30b7", "\u30a7"), ("\u30b8", "\u30a7"), ("\u30c1", "\u30a7"),                         # シェ ジェ チェ
    ("\u30c6", "\u30a3"), ("\u30c6", "\u30e5"), ("\u30c7", "\u30a3"), ("\u30c7", "\u30e5"),   # ティ テュ ディ デュ
    ("\u30c8", "\u30a5"), ("\u30c9", "\u30a5"),                                               # トゥ ドゥ
    ("\u30c4", "\u30a1"), ("\u30c4", "\u30a3"), ("\u30c4", "\u30a7"), ("\u30c4", "\u30a9"),   # ツァ ツィ ツェ ツォ
}

# language decisions this song needs (everything else follows from the script).
# The Chinese material is exactly one 6-glyph run; the kanji-only tokens are all
# obviously Japanese (Chinese uses 我, never 私).
DEFAULT_ZH_RUNS = ["\u6211\u60f3\u8981\u7684\u4e00\u5207"]                    # 我想要的一切
DEFAULT_JA_TOKENS = ["\u79c1\u4ee5\u5916", "\u4eca", "\u79c1"]                 # 私以外, 今, 私

VOWEL_GROUP = re.compile(r"[aeiouy]+")


def is_hira(c: str) -> bool:
    return 0x3041 <= ord(c) <= 0x3096


def is_kata(c: str) -> bool:
    return 0x30A1 <= ord(c) <= 0x30FA


def is_kana(c: str) -> bool:
    return is_hira(c) or is_kata(c) or c == LONG


def is_cjk(c: str) -> bool:
    return 0x3400 <= ord(c) <= 0x4DBF or 0x4E00 <= ord(c) <= 0x9FFF


def is_latin(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or c == "'"


def kind_of(c: str) -> str:
    if is_kana(c):
        return "kana"
    if is_cjk(c):
        return "cjk"
    if is_latin(c):
        return "latin"
    return "other"


def runs_of(tok: str) -> list[str]:
    """Maximal runs of one script class.  Punctuation and digits are 'other'."""
    out: list[str] = []
    for c in tok:
        if out and kind_of(out[-1][0]) == kind_of(c):
            out[-1] += c
        else:
            out.append(c)
    return out


def latin_syllables(w: str) -> int:
    """Vowel-group heuristic; flagged as tentative and refined from onsets later."""
    s = w.lower()
    n = len(VOWEL_GROUP.findall(s))
    if s.endswith("e") and not s.endswith(("le", "ee", "ie")) and len(s) > 3 and n > 1:
        n -= 1
    return max(1, n)


@dataclass
class Unit:
    """One sung glyph group == one entry of Word.syl."""
    text: str
    morae: float
    lo: float
    hi: float
    cls: str                       # kana | cjk | latin | punct
    lang: str                      # ja | zh | en | --
    tentative: bool = False        # weight is a prior/guess, onsets refine it

    def bump(self, extra: float) -> None:
        self.morae += extra
        self.lo += extra
        self.hi += extra

    def as_dict(self) -> dict:
        return {"u": self.text, "morae": round(self.morae, 3), "lo": self.lo,
                "hi": self.hi, "cls": self.cls, "lang": self.lang,
                "tentative": self.tentative}

# ------------------------------------------------------------------- tokenizing
def token_units(tok: str, ja_tokens: set[str], zh_runs: set[str],
                unresolved: list[tuple[str, str]]) -> list[Unit]:
    """Split one whitespace token into sung units (== the entries of Word.syl)."""
    has_kana = any(is_kana(c) for c in tok)
    default = "ja" if (has_kana or tok in ja_tokens) else "zh"
    units: list[Unit] = []
    pending = ""                      # leading punctuation waits for its glyph
    for run in runs_of(tok):
        k = kind_of(run[0])
        if k == "latin":
            s = latin_syllables(run)
            units.append(Unit(pending + run, s, s, s, "latin", "en", True))
            pending = ""
        elif k == "cjk":
            lang = "zh" if run in zh_runs else default
            if not has_kana and tok not in ja_tokens and run not in zh_runs:
                unresolved.append((tok, run))       # reported, never silently guessed
            for i, c in enumerate(run):
                head = pending if i == 0 else ""
                if lang == "zh":
                    units.append(Unit(head + c, 1, 1, 1, "cjk", "zh", False))
                else:
                    units.append(Unit(head + c, 2, 1, 3, "cjk", "ja", True))
                pending = ""
        elif k == "kana":
            for i, c in enumerate(run):
                prev = units[-1].text[-1] if units else ""
                head = pending if i == 0 else ""
                if c in DIG0 and prev and is_kana(prev):
                    units[-1].text += c                     # digraph: no extra mora
                elif c in SMLV and prev and is_kana(prev):
                    units[-1].text += c
                    if (prev, c) not in KATA_DIGRAPH:
                        units[-1].bump(1)                   # long vowel: one mora
                else:
                    units.append(Unit(head + c, 1, 1, 1, "kana", "ja", False))
                    pending = ""
        else:                                               # punctuation, digits, 々
            if units:
                units[-1].text += run                       # 々 inherits the weight
            else:
                pending += run
    if pending:
        if units:
            units[-1].text += pending
        else:
            units.append(Unit(pending, 0, 0, 0, "punct", "--", False))
    return units


@dataclass
class LineUnits:
    i: int
    text: str
    tokens: list[str] = field(default_factory=list)
    units: list[list[Unit]] = field(default_factory=list)

    @property
    def all_units(self) -> list[Unit]:
        return [u for us in self.units for u in us]

    @property
    def n_units(self) -> int:
        return sum(len(us) for us in self.units)

    def morae(self) -> tuple[float, float, float]:
        us = self.all_units
        return (sum(u.lo for u in us), sum(u.morae for u in us), sum(u.hi for u in us))

    def langs(self) -> set[str]:
        return {u.lang for u in self.all_units if u.lang != "--"}


def read_lines(path=None) -> list[str]:
    p = path or common.LYRICS_TXT
    return [l for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def parse(ja_tokens=None, zh_runs=None, lines=None):
    """-> (lines, unresolved, ja_tokens, zh_runs)"""
    lines = lines if lines is not None else read_lines()
    ja = set(DEFAULT_JA_TOKENS if ja_tokens is None else ja_tokens)
    zh = set(DEFAULT_ZH_RUNS if zh_runs is None else zh_runs)
    unresolved: list[tuple[str, str]] = []
    out: list[LineUnits] = []
    for i, text in enumerate(lines):
        lu = LineUnits(i, text, text.split(" "))
        lu.units = [token_units(t, ja, zh, unresolved) for t in lu.tokens]
        out.append(lu)
    return out, unresolved, sorted(ja), sorted(zh)


# ------------------------------------------------------------------ invariants
def check(lines: list[LineUnits]) -> list[str]:
    """The invariants the engine relies on.  Empty list == clean."""
    bad: list[str] = []
    for lu in lines:
        if " ".join(lu.tokens) != lu.text:
            bad.append(f"L{lu.i:02d}: ' '.join(words) != text")
        for tok, us in zip(lu.tokens, lu.units):
            joined = "".join(u.text for u in us)
            if joined != tok:
                bad.append(f"L{lu.i:02d}: units rebuild {joined!r} != token {tok!r}")
            for u in us:
                if not u.text:
                    bad.append(f"L{lu.i:02d}: empty unit in {tok!r}")
                elif u.morae <= 0 and u.cls != "punct":
                    bad.append(f"L{lu.i:02d}: zero-mora {u.cls} unit {u.text!r} in {tok!r}")
    return bad


# ---------------------------------------------------------------------- reports
def table(lines: list[LineUnits], unresolved: list, ja: list, zh: list) -> str:
    o: list[str] = []
    w = max(len(u.text) for lu in lines for u in lu.all_units)
    for lu in lines:
        lo, prior, hi = lu.morae()
        langs = ",".join(sorted(lu.langs())) or "--"
        o.append(f"L{lu.i:02d}  [{langs}]  units {lu.n_units:3d}   morae {lo:6.1f} : {prior:6.1f} : {hi:6.1f}")
        o.append(f"      {lu.text}")
        for tok, us in zip(lu.tokens, lu.units):
            glyphs = "  ".join(f"{u.text:<{w}}:{u.morae:g}{'?' if u.tentative else ''}" for u in us)
            o.append(f"        {tok!r}  ->  {glyphs}   [{sum(u.lo for u in us):g}..{sum(u.hi for u in us):g}]")
        o.append("")
    o.append("language decisions")
    o.append(f"  zh runs used   : {zh}")
    o.append(f"  ja tokens used : {ja}")
    if unresolved:
        o.append(f"  UNRESOLVED CJK-only tokens ({len(unresolved)}) -- add to ja_tokens or zh_runs:")
        for tok, run in unresolved:
            o.append(f"     {tok!r} / run {run!r}")
    else:
        o.append("  unresolved CJK-only tokens: none (every run decided by script or by list)")
    o.append("  marker: '?' = weight is a prior (ja kanji 1..3, latin heuristic);")
    o.append("          the onsets in L6 decide the exact split inside [lo..hi].")
    return "\n".join(o)


def totals(lines: list[LineUnits], vocal_seconds: float = 221.0) -> str:
    n = len(lines)
    units = sum(lu.n_units for lu in lines)
    lo = sum(u.lo for lu in lines for u in lu.all_units)
    prior = sum(u.morae for lu in lines for u in lu.all_units)
    hi = sum(u.hi for lu in lines for u in lu.all_units)
    cls = Counter(u.cls for lu in lines for u in lu.all_units)
    kana_m = sum(u.morae for lu in lines for u in lu.all_units if u.cls == "kana")
    tent = sum(1 for lu in lines for u in lu.all_units if u.tentative)
    return "\n".join([
        "totals over %d line occurrences (%d unique texts)" % (n, len(set(lu.text for lu in lines))),
        "  units            %5d   (%.1f per line)" % (units, units / n),
        "  by class         " + "  ".join(f"{k}:{v}" for k, v in sorted(cls.items())),
        "  kana morae       %5.0f   (exact: orthography only)" % kana_m,
        "  morae lo : prior : hi   %.0f : %.0f : %.0f" % (lo, prior, hi),
        "  per second over %.0f s of vocal activity   units %.2f   morae %.2f   range %.2f-%.2f"
        % (vocal_seconds, units / vocal_seconds, prior / vocal_seconds, lo / vocal_seconds, hi / vocal_seconds),
        "  tentative units  %5d of %d   (%.0f%% of the weights come from the onsets)"
        % (tent, units, 100 * tent / units),
        "  note: vocal_seconds=%.0f is the probe figure; L2 re-measures it and replaces this line."
        % vocal_seconds,
    ])


# ------------------------------------------------------------------ ground truth
def build_src(lines: list[LineUnits], ja: list, zh: list, path=None) -> dict:
    """data/lyrics.src.json: the authored ground truth the aligner must honour."""
    occ = Counter(read_lines())
    repeats = [{"text": t, "in_file": c} for t, c in sorted(occ.items(), key=lambda x: -x[1]) if c > 1]
    doc = {
        "source": "audio/lyrics.txt",
        "note": ("Ground truth for lyric alignment.  Line order IS performance order: the "
                 "repeats are already written out in the file (only the section-specific "
                 "wording differs), so the %d lines are the %d sung line occurrences. "
                 "Expanding the file's own duplicates again would double-count them -- add "
                 "to extra_occurrences only a repeat the audio proves the sheet omits. "
                 "anchor / morae_override are the hand-fix slots." % (len(lines), len(lines))),
        "occurrence_model": {
            "order": "file order",
            "n_occurrences": len(lines),
            "n_unique_texts": len(set(l.text for l in lines)),
            "repeats_in_file": repeats,
            "extra_occurrences": [],
        },
        "language": {
            "rule": ("per run: a CJK run inside a token that contains kana is Japanese unless "
                     "listed in zh_runs; a token with no kana is Japanese iff listed in "
                     "ja_tokens, otherwise Chinese -- and reported, never assumed"),
            "ja_tokens": ja,
            "zh_runs": zh,
        },
        "offsets": {"offset_ms": common.OFFSET_MS},
        "lines": [{"i": lu.i, "text": lu.text, "words": lu.tokens,
                   "anchor": None, "morae_override": None} for lu in lines],
    }
    if path:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return doc


def main(argv: list[str]) -> int:
    what = argv[1] if len(argv) > 1 else "all"
    lines, unresolved, ja, zh = parse()
    bad = check(lines)
    if what in ("table", "all"):
        print(table(lines, unresolved, ja, zh))
        print()
        print(totals(lines))
        (common.QA / "units_table.txt").write_text(
            table(lines, unresolved, ja, zh) + "\n\n" + totals(lines) + "\n", encoding="utf-8")
        print("\nwrote %s" % (common.QA / "units_table.txt"))
    if what in ("src", "all"):
        build_src(lines, ja, zh, common.DATA / "lyrics.src.json")
        print("wrote %s" % (common.DATA / "lyrics.src.json"))
    if what in ("check", "all"):
        print("\ninvariants: %s" % ("CLEAN" if not bad else "%d FAILURES" % len(bad)))
        for b in bad[:40]:
            print("   " + b)
        print("unresolved language runs: %d" % len(unresolved))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

