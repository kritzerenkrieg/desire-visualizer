# desire-visualizer

Syllable-level lyric timing pipeline for the DESIRE song, plus a pilot renderer that
shows the music-to-syllable mapping so it can be judged by eye and ear. The output
(`data/lyrics.approx.json`) follows the engine schema of
`reference/pdoom-video/app/src/engine/lyrics.ts` (100 fps envelope contract,
`analysis time == mux time`, `offset_ms = 0`).

## Layout

| Path | What |
|---|---|
| `audio/` | inputs: `source.mp3` (fMP4/AAC despite the name), `vocals.mp3` (hand-made vocal isolation), `lyrics.txt` (display text, one line per sung line), `syllable.txt` (sung syllable sequence per line) |
| `analysis/` | the active pipeline: `analyze.py`, `word_syllables.py`, `pilot.py`, `common.py` |
| `legacy/` | the superseded mora/DP pass (`lyric_units.py`, `align_lyrics.py`) and the L1 decode proof (`check_decode.py`). `analyze.py` / `pilot.py` still import the orthography helpers from here. |
| `data/` | `lyrics.approx.json` (the alignment the engine reads), `syllable_map.json` (editable syllable→word grouping) |
| `analysis/work/` | intermediates: `features_{mix,stem,vocal}.npz`, whisper word-times cache, decode report |
| `analysis/qa/` | reports and QA plots |
| `out/` | pilot renders `pilot_<a>_<b>.mp4` + `pilot_<a>_<b>_strip.png` |
| `tools/ff/` | static ffmpeg (no root needed) |

All commands run from the repo root with the project venv:

```sh
.venv/bin/python analysis/<script>.py ...
```

## Prerequisites

* Python 3.13 venv at `.venv/` with numpy, scipy, matplotlib, pillow, fonttools,
  stable-ts, faster-whisper (torch with CUDA optional; whisper falls back to CPU)
* `tools/ff/ffmpeg` (checked in; the scripts prepend `tools/ff` to `PATH` themselves)
* a CJK font — the pilot scans system fonts and uses `analysis/work/fonts/` as fallback

## Operating the pipeline

Four stages, in order. Each one reuses its cache/artifacts, so re-runs are cheap
unless you pass `--force`.

### 1. Features (L2)

```sh
.venv/bin/python analysis/analyze.py --vocal     # features_vocal.npz  (used by word timing)
.venv/bin/python analysis/analyze.py --stems     # features_stem.npz   (default pilot --src)
.venv/bin/python analysis/analyze.py             # features_mix.npz
```

Writes `analysis/work/features_<src>.npz` (envelopes at 100 fps, note onsets,
beat grid: bpm / period / phase) and QA plots + `analysis/qa/l2_report_<src>.txt`.
`--alloc` additionally writes a v0 `data/lyrics.approx.json` (mora count guess only —
superseded by stage 3, don't use).

### 2. Syllable map (draft grouping)

```sh
.venv/bin/python analysis/word_syllables.py map
```

Regenerates `data/syllable_map.json`: which sung syllable token belongs to which
display word. This is a *draft*; edit the JSON before aligning — moving a token
between neighbouring words is the intended fix for wrong groupings.

### 3. Word timing + syllable spans → `data/lyrics.approx.json`

```sh
.venv/bin/python analysis/word_syllables.py align --model large-v3-turbo
```

* word times: whisper on the vocal stem, matched character-wise onto the known
  `lyrics.txt` text (`--engine {auto,faster-whisper,openai}`, `--device {cuda,cpu}`,
  `--locate {transcribe,align}`). Word times are cached in
  `analysis/work/word_times_<model>.json`; `--force` re-transcribes.
* syllable cuts: `data/syllable_map.json` grouping, cut points snapped to the BPM
  lattice (`--subdiv 2` = 1/8 notes; `--bpm/--phase/--period` override the grid).

Outputs `data/lyrics.approx.json` + `analysis/qa/word_syllables_report.txt`.

### 4. Verify + render

```sh
.venv/bin/python analysis/word_syllables.py check          # engine-schema invariants
.venv/bin/python analysis/pilot.py                          # 60-120 s smoke render
.venv/bin/python analysis/pilot.py --start 55 --end 545 --fps 20   # full song
```

The pilot writes `out/pilot_<a>_<b>.mp4` (line + wipe + syllable timeline with the
audio muxed at exactly `--start`) and `out/pilot_<a>_<b>_strip.png` (static score
strip). Flags: `--src {stem,mix,vocal}` picks the features npz for the presence /
onset backdrop, `--data PATH` renders a candidate alignment without touching the
adopted one, `--no-video` writes the strip only, `--font PATH` forces a face.
The wipe is a port of the engine's `Lyrics.lineCharProgress` / `Lyrics.wordProgress`,
so what the pilot shows is what the app will draw for the same data.

Full-song at 20 fps renders at roughly real-time x10 on this host and prints
progress every few hundred frames; expect several minutes for 900 s of audio.

## Reviewing the result

Watch `out/pilot_<a>_<b>.mp4` with sound; the important observations are:

* does each syllable land on the note you hear? (amber block = active syllable)
* does the wipe reach the end of the line at the same time as the singing?
* silences / chops ≥ 0.35 s are marked; a syllable starting right after one is
  where alignment errors are most likely.

Known weak spots in the current data: the outro chant (line 68, ~550 s) is
mis-timed (whisper limitation) and 55 lines in `analysis/qa/word_syllables_report.txt`
have an out-of-range duration/mora ratio — both visible in the full render.

## Legacy

`legacy/` holds the superseded orthography→mora DP pass:

* `lyric_units.py` – script-native mora/syllable weights from the text
* `align_lyrics.py` – DP match of morae to note onsets → v2 `lyrics.approx.json`
* `check_decode.py` – L1 two-decoder cross-check (validates `OFFSET_MS = 0`)

They run standalone (`sys.path` shims resolve `analysis/common.py`) but nothing
in the active pipeline executes them anymore: `analyze.py` and `pilot.py` only
import `lyric_units` / `align_lyrics` for the orthography text helpers. Their
outputs in `analysis/work/` (`lyrics.approx.legacy_v2.json` etc.) are kept for
comparison only.

## Provenance notes

* `common.py` carries the container facts of `audio/source.mp3` (fMP4/AAC, no edit
  list, no priming trim) — that is why the analysis timeline equals playback time
  and no offset correction is applied anywhere.
* `data/lyrics.approx.json` `method`/`source` fields record the exact flags and
  features npz used for the adopted run.