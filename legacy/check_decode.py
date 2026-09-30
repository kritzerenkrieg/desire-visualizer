#!/usr/bin/env python3
"""L1: prove what the decode is, and that two independent decoders agree.

The engine muxes the same file (`-i audio/source.mp3` in app/scripts/render.ts) that
this analysis decodes, so as long as both start at the same pts the analysis timeline
IS the playback timeline and offset_ms stays 0.  The source is a fragmented MP4 with no
`elst` edit list and sidx.earliest_presentation_time = 0, so no decoder trims the AAC
priming; this script measures that instead of trusting it.

Writes analysis/work/decode_report.json.
Run:  .venv/bin/python analysis/check_decode.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))  # legacy/: shared lib lives in analysis/

import numpy as np

import common


def ffmpeg_decode(path, channels, rate):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(-1, channels) if channels > 1 else a


def pyav_decode(path):
    import av
    c = av.open(str(path))
    s = c.streams.audio[0]
    meta = {
        "codec": s.codec_context.name,
        "rate": s.codec_context.sample_rate,
        "channels": s.codec_context.channels,
        "start_time": float(s.start_time or 0),
        "container_duration": float(c.duration / av.time_base) if c.duration else None,
        "time_base": str(s.time_base),
    }
    parts, frames = [], 0
    for fr in c.decode(s):
        try:
            a = fr.to_ndarray(format="fltp")
        except Exception:
            a = fr.to_ndarray()
        if a.ndim == 1:
            a = a[None, :]
        parts.append(a)
        frames += 1
    y = np.concatenate(parts, axis=1).T.astype(np.float32)   # (samples, channels)
    meta["frames"] = frames
    meta["samples"] = int(y.shape[0])
    meta["duration"] = float(y.shape[0] / meta["rate"])
    return meta, y


def lag_and_diff(a, b, window=44100 * 6, coarse=4096, step=16):
    """Directly measure the sample offset between two decodes.

    Both decoders are FFmpeg underneath, so at the true offset the difference is float
    rounding and at any other offset it is O(signal): a coarse sweep plus a unit-step
    refine finds the offset unambiguously (it would separate 0 from a 1024-sample priming).
    """
    n = min(len(a), len(b))
    ma = a[:n].sum(1) if a.ndim > 1 else a[:n]
    mb = b[:n].sum(1) if b.ndim > 1 else b[:n]
    t0 = 44100 * 10
    ref = ma[t0: t0 + window].astype(np.float64)

    def diff_at(lag):
        lo, hi = t0 + lag, t0 + lag + window
        if lo < 0 or hi > n:
            return float("inf")
        return float(np.abs(mb[lo:hi] - ref).mean())

    coarse_lags = range(-coarse, coarse + 1, step)
    best = min(((diff_at(l), l) for l in coarse_lags), key=lambda x: x[0])
    fine = [l for l in range(best[1] - step, best[1] + step + 1)]
    best = min(((diff_at(l), l) for l in fine), key=lambda x: x[0])
    return best[1], best[0], diff_at(0)


def activity(y, rate, thresh_db=-60.0):
    """First/last time above a dBFS threshold relative to the peak, plus active seconds."""
    hop = rate // 100
    n = len(y) // hop
    e = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(e + 1e-20)
    peak = db.max()
    out = {"peak_dbfs": float(peak)}
    for t in (thresh_db, -50.0, -40.0, -30.0):
        idx = np.where(db > peak + t)[0]
        out[f"above_{abs(int(t))}db"] = {
            "first_s": round(float(idx[0] / 100), 3) if len(idx) else None,
            "last_s": round(float(idx[-1] / 100), 3) if len(idx) else None,
            "seconds": round(float(len(idx) / 100), 3),
        }
    return out


def main() -> int:
    rep: dict = {"source": str(common.AUDIO.name), "container": common.CONTAINER}
    meta, y = pyav_decode(common.AUDIO)
    rep["pyav"] = meta

    mono = ffmpeg_decode(common.WORK / "mix22050.f32", 1, common.SR_FEAT)
    st = ffmpeg_decode(common.WORK / "mix44100.f32", 2, common.SR)
    rep["ffmpeg"] = {"mono22050_samples": int(mono.size), "stereo44100_frames": int(st.shape[0])}

    exp_m = round(common.CONTAINER["samples"] * common.SR_FEAT / common.SR)
    rep["checks"] = {
        "stereo_frames_eq_container_samples": bool(st.shape[0] == common.CONTAINER["samples"]),
        "mono_samples_eq_expected": bool(mono.size == exp_m),
        "pyav_samples_eq_ffmpeg": bool(meta["samples"] == st.shape[0]),
        "pyav_frames_eq_container_frames": bool(meta["frames"] == common.CONTAINER["frames"]),
    }

    lag, mad, maxd = lag_and_diff(y.astype(np.float32), st)
    rep["decoder_agreement"] = {
        "offset_samples": int(lag),
        "offset_ms": round(lag / common.SR * 1000, 4),
        "mean_abs_diff_on_window": float(mad),
        "max_abs_diff": maxd,
    }
    rep["activity"] = activity(mono.mean(1) if mono.ndim > 1 else mono, common.SR_FEAT)
    rep["offset_ms_used"] = common.OFFSET_MS

    print(json.dumps(rep, indent=1))
    (common.WORK / "decode_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")

    ok = all(rep["checks"].values()) and lag == 0
    print("\nL1: %s" % ("PASS (decode == container to the sample, decoders agree at lag 0)"
                        if ok else "FAIL -- inspect decode_report.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
