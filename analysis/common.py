"""Shared paths and provenance constants for the desire-visualizer analysis.

Import this FIRST (before numpy / pyav / torch) so caches land in analysis/.cache/
and matplotlib never writes to $HOME.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # desire-visualizer/
ANALYSIS = ROOT / "analysis"
DATA = ROOT / "data"
AUDIO = ROOT / "audio" / "source.mp3"                  # fMP4/AAC despite the .mp3 name
VOCALS = ROOT / "audio" / "vocals.mp3"                 # a hand-made vocal isolation to work from
LYRICS_TXT = ROOT / "audio" / "lyrics.txt"
SYLLABLE_TXT = ROOT / "audio" / "syllable.txt"       # sung syllable sequence per line
QA = ANALYSIS / "qa"                                   # plots and gate artifacts
WORK = ANALYSIS / "work"                               # decode + intermediates
OUT = ROOT / "out"                                     # renders (mp4/png the pilot writes)
CACHE = ANALYSIS / ".cache"                            # model / matplotlib caches
TOOLS = ROOT / "tools"                                 # static ffmpeg (no root on this host)

for _var, _sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", "xdg")]:
    os.environ.setdefault(_var, str(CACHE / _sub))
for _d in (QA, WORK, CACHE, DATA, OUT):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- timeline
# SR = whatever the mux path uses; SR_FEAT = the feature/decode rate.
SR = 44100
SR_FEAT = 22050
FPS = 100                                              # envelope rate promised to the engine

# Our time reference is a plain ffmpeg decode of AUDIO, and so is the engine's
# muxed audio (`-i audio/source.mp3` in app/scripts/render.ts).  The file is a
# fragmented MP4 (ftyp dash/iso6mp4, "produced by Google") carrying one AAC track
# at 44100 Hz: 25874 frames x 1024 samples = 26494976 samples = 600.7931066 s,
# which is exactly the mvhd/mdhd duration.  There is no `elst` edit list and
# `sidx.earliest_presentation_time` is 0, so every decoder (ffmpeg, PyAV, Chrome)
# starts at the same pts with the same ~1024-sample AAC priming included.
# Nothing is trimmed anywhere => analysis time == mux time, so this stays 0.0
# unless L1's two-decoder cross-correlation says otherwise.  (Upstream's -23.0 ms
# was a different failure: a LAME mp3 whose encoder delay the stem decode did not
# trim -- see pdoom-video/analysis/common.py.)
OFFSET_MS = 0.0

# Provenance of the container, measured (analysis/work/source.info.json).
CONTAINER = {
    "brand": "dash / iso6mp4",
    "codec": "aac",
    "sample_rate": SR,
    "frames": 25874,
    "samples_per_frame": 1024,
    "samples": 26494976,
    "duration_s": 600.7931065759637,
    "edit_list": None,
    "priming_samples": 1024,
}
