#!/usr/bin/env python3
"""L2: audio features, the phrase map, and v0 lyric times for the new song.

Owns its DSP (~150 lines) instead of pulling librosa: numba is the riskiest dependency on
Python 3.13 and nothing here needs it.  The envelope recipe IS upstream's, because
data/audio.json's engine contract documents it -- a 46 ms RMS window, a one-pole follower
with 10 ms attack / 90 ms release, divided by the 99th percentile and clipped to 0..1.

Everything runs at 44100 Hz with HOP = 441 samples, so one frame is exactly 10 ms and
frame i is centred on i/100 s.  (At 22050 the hop would be 220.5 samples: the frame rate
would be 100.23 fps and the envelopes would drift ~1.4 s across the song.)

bands  low < 150 Hz, mid 150-2000 Hz, high > 4 kHz (Butterworth 4, zero-phase).
vocal  Tier A, no stems: an HPSS-lite mask (median over time -> harmonic, over frequency
       -> percussive) restricted to 200-4000 Hz gives a vocal-presence envelope; the
       spectral flatness of that band gives harmonicity (voicing); positive log-mel flux
       gives note onsets.  Tier B runs the identical functions on the Demucs vocals stem.

Run:  .venv/bin/python analysis/analyze.py             # features + plots
      .venv/bin/python analysis/analyze.py --alloc     # + data/lyrics.approx.json (v0)
      .venv/bin/python analysis/analyze.py --stems     # vocal stem instead of the mix
      .venv/bin/python analysis/analyze.py --vocal     # the isolation in audio/vocals.mp3
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

import common

sys.path.append(str(common.ROOT / "legacy"))  # mora/orthography helpers live in legacy/
import lyric_units

FPS = common.FPS                    # 100 envelopes per second (the engine's contract)
SR = common.SR                      # 44100
HOP = SR // FPS                     # 441 samples == exactly 10 ms
N_FFT = 2048                        # 46.4 ms window, 21.5 Hz bins
WIN_RMS = 2048                      # upstream's window
assert SR % FPS == 0

BAND_LOW = (None, 150.0)
BAND_MID = (150.0, 2000.0)
BAND_HIGH = (4000.0, None)
VOCAL_BAND = (200.0, 4000.0)
ONSET_FLOOR = 0.05                  # vocal presence (normalised) a note onset needs
NOTE_BUDGET = 1.3                   # detector may keep this many notes per mora of the text

N_MELS = 64
MEL_FMIN, MEL_FMAX = 100.0, 8000.0

STEMS = common.ANALYSIS / "stems" / "htdemucs_ft" / "source44100"
FFMPEG = common.TOOLS / "ff" / "ffmpeg"
OFFSET_SEARCH_S = 0.4               # how far a foreign stem is searched for its own delay
OFFSET_GRID_MS = 1.0                # resolution of that search


def n_frames(n_samples: int) -> int:
    return int(round(n_samples / SR * FPS))


def read_mono() -> np.ndarray:
    """The 44.1 kHz decode as mono float32 (mean of the two channels)."""
    x = np.fromfile(common.WORK / "mix44100.f32", dtype="<f4")
    return x.reshape(-1, 2).mean(1).astype(np.float32)


def frame_rms(x, win=WIN_RMS, sr=SR, fps=FPS) -> np.ndarray:
    """RMS at `fps`, frame i centred on i/fps (upstream's recipe, cumulative-sum)."""
    hop = sr / fps
    n = n_frames(len(x)) if (sr, fps) == (SR, FPS) else int(np.ceil(len(x) / sr * fps))
    pad = np.pad(np.asarray(x, dtype=np.float64), (win // 2, win // 2 + int(hop) + 2))
    idx = (np.arange(n) * hop).astype(int)
    c = np.concatenate([[0.0], np.cumsum(pad ** 2)])
    return np.sqrt(np.maximum((c[idx + win] - c[idx]) / win, 0)).astype(np.float32)


def smooth_env(x, fps=FPS, attack=0.010, release=0.090) -> np.ndarray:
    """One-pole follower: fast attack, slower release (visually pleasing, upstream's)."""
    aa = np.exp(-1.0 / (attack * fps))
    ar = np.exp(-1.0 / (release * fps))
    y = np.empty_like(x, dtype=np.float32)
    s = 0.0
    for i, v in enumerate(x):
        a = aa if v > s else ar
        s = a * s + (1.0 - a) * v
        y[i] = s
    return y


def norm01(x, pct=99.0) -> np.ndarray:
    ref = np.percentile(x, pct)
    return np.clip(x / (ref + 1e-12), 0, 1).astype(np.float32)


def band_sos(lo, hi, sr=SR):
    if lo and hi:
        return butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    if hi:
        return butter(4, hi, btype="low", fs=sr, output="sos")
    return butter(4, lo, btype="high", fs=sr, output="sos")


def band_rms(x, lo, hi, sr=SR, fps=FPS) -> np.ndarray:
    y = sosfiltfilt(band_sos(lo, hi, sr), np.asarray(x, dtype=np.float64))
    return frame_rms(y, fps=fps, sr=sr)


FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SR)
VOC_BINS = (FREQS >= VOCAL_BAND[0]) & (FREQS <= VOCAL_BAND[1])


def stft_mag(x, block=8192) -> np.ndarray:
    """|STFT| (bins, frames); frame i is centred on i/fps, phase discarded."""
    x = np.asarray(x, dtype=np.float32)
    xp = np.pad(x, (N_FFT // 2, N_FFT // 2 + HOP))
    n = n_frames(len(x))
    view = np.lib.stride_tricks.sliding_window_view(xp, N_FFT)[::HOP][:n]
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((N_FFT // 2 + 1, n), dtype=np.float32)
    for s in range(0, n, block):
        e = min(n, s + block)
        out[:, s:e] = np.abs(np.fft.rfft(view[s:e] * win, axis=1)).T
    return out


def hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_matrix(n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX) -> np.ndarray:
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    M = np.zeros((n_mels, len(FREQS)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        M[i] = np.clip(np.minimum((FREQS - lo) / (c - lo), (hi - FREQS) / (hi - c)), 0, None)
    return M


def hpss_mask(Sc, kt=17, kf=17) -> np.ndarray:
    """Harmonic/percussive soft mask of a magnitude spectrogram (median in time vs freq)."""
    H = median_filter(Sc, size=(kf, 1), mode="nearest")
    P = median_filter(Sc, size=(1, kt), mode="nearest")
    return ((H * H) / (H * H + P * P + 1e-12)).astype(np.float32)


def vocal_features(S, mel):
    """-> (presence, onset strength, flatness) envelopes at 100 fps.

    presence : harmonic energy inside VOCAL_BAND -- the Tier A vocal proxy
    strength : positive log-mel flux over the full band -- the note-onset signal
    flatness : spectral flatness inside VOCAL_BAND -- low means harmonic (voiced)

    `strength` is a *relative* measure (a difference of logs), so where the stem is silent it
    reports the noise floor's own fluctuations.  Measured on the stem, the instrumental intro
    yielded onsets as strong as singing and a silent stretch between phrases yielded stronger
    ones than the notes beside it, which is why every caller gates it by the absolute level
    (`presence`) before peak-picking -- see ONSET_FLOOR.
    """
    Sc = S[VOC_BINS]
    harm = Sc * hpss_mask(Sc)
    presence = np.sqrt((harm * harm).sum(0)).astype(np.float32)
    logmel = np.log(mel @ S + 1e-10)
    flux = np.diff(logmel, axis=1, prepend=logmel[:, :1])
    strength = np.maximum(flux, 0).mean(0).astype(np.float32)
    flat = (np.exp(np.log(Sc + 1e-10).mean(0)) / (Sc.mean(0) + 1e-10)).astype(np.float32)
    return presence, strength, flat


def peaks(strength, thresh, min_sep_frames=4):
    """Local maxima above `thresh` at least `min_sep_frames` apart."""
    idx, props = find_peaks(strength, height=thresh, distance=min_sep_frames)
    return idx, props["peak_heights"].astype(np.float32)


def onset_peaks(strength, target=None, floor_pct=85.0, min_sep_frames=3):
    """Peaks of the gated flux, crowded down to a plausible note count.

    One global threshold cannot serve a whole song: the 95th percentile of the loud frames is
    too strict for a quiet verse and too lax for a fast chorus, so it both loses real notes and
    invents them.  Take every local maximum above a permissive threshold instead, then resolve
    crowding by dropping the weaker of the closest pair until at most `target` notes remain.  A
    dense passage loses its weakest candidates (the tempo forbids them being separate notes
    anyway) while a sparse passage keeps every one of its.
    """
    loud = strength[strength > 0]
    thr = float(np.percentile(loud, floor_pct)) if loud.size else 1.0
    idx, h = peaks(strength, thr, min_sep_frames=min_sep_frames)
    idx, h = list(idx), list(h)
    while target and len(idx) > target:
        k = int(np.argmin(np.diff(idx)))
        drop = k if h[k] <= h[k + 1] else k + 1
        idx.pop(drop), h.pop(drop)
    return np.array(idx, dtype=np.int64), np.array(h, dtype=np.float32)


def nearest_onset(t, ot, lo, hi, tol):
    """Snap t to the closest note onset within tol, staying inside (lo, hi)."""
    if len(ot) == 0:
        return t, False
    i = int(np.searchsorted(ot, t))
    cands = [ot[j] for j in (i - 1, i) if 0 <= j < len(ot)]
    best = min(cands, key=lambda c: abs(c - t))
    if abs(best - t) <= tol and lo + 0.02 < best < hi - 0.02:
        return float(best), True
    return t, False


def split_word(units, t0, t1, ot, tol):
    """Distribute a word's span across its units by mora, snapped to nearby onsets."""
    w = np.array([u.morae for u in units], dtype=float)
    cw = np.cumsum(w)
    spans, prev = [], t0
    for k in range(len(units)):
        if k == len(units) - 1:
            end = t1
        else:
            target = t0 + (t1 - t0) * cw[k] / cw[-1]
            end, _ = nearest_onset(target, ot, prev, t1, tol)
            end = min(max(end, prev + 0.02), t1 - 0.02 * (len(units) - k - 1))
        spans.append((round(prev, 3), round(end, 3)))
        prev = end
    return spans


def allocate(lines, morae, runs, onsets, gap_snap=0.6, unit_snap=0.08,
             min_line=0.25, max_line=12.0):
    """v0: order-constrained proportional walk over the active (non-silent) timeline.

    Monotonic by construction.  Each phrase run contributes capacity proportional to
    duration x (0.5 + its energy), so a quiet breathy run holds fewer morae than a loud
    one; boundaries that land near a silence are snapped into it, and unit boundaries
    inside a line are snapped to the nearest note onset.  The onset-capacity DP is the
    v1 upgrade once the stem-based onsets are in.
    """
    segs = [(a, b, max(1e-3, (b - a) * (0.5 + min(1.0, m)))) for a, b, m, _ in runs]
    total_cap = sum(w for _, _, w in segs)
    cap_cum = np.concatenate([[0.0], np.cumsum([w for _, _, w in segs])]) / total_cap
    total_dem = float(sum(morae))
    dem_cum = np.concatenate([[0.0], np.cumsum(morae)]) / total_dem
    ot = onsets[0]

    def time_at(f):
        k = int(np.clip(np.searchsorted(cap_cum, f) - 1, 0, len(segs) - 1))
        local = (f - cap_cum[k]) / (cap_cum[k + 1] - cap_cum[k] + 1e-12)
        a, b, _ = segs[k]
        return a + local * (b - a)

    bounds = [time_at(f) for f in dem_cum]
    bounds[0], bounds[-1] = segs[0][0], segs[-1][1]
    gaps = [0.5 * (runs[i][1] + runs[i + 1][0]) for i in range(len(runs) - 1)]
    snapped = [False] * len(bounds)
    for i in range(1, len(bounds) - 1):
        near = min(gaps, key=lambda g: abs(g - bounds[i]))
        if abs(near - bounds[i]) <= gap_snap:
            bounds[i], snapped[i] = near, True
    for i in range(1, len(bounds)):
        bounds[i] = max(bounds[i], bounds[i - 1] + min_line)
    bounds[-1] = max(bounds[-1], bounds[-2] + min_line)

    rate = total_dem / max(1e-6, sum(b - a for a, b, _ in segs))
    out, rep = [], []
    for i, lu in enumerate(lines):
        t0, t1 = bounds[i], bounds[i + 1]
        spans = [split_word(us, t0, t1, ot, unit_snap) for us in lu.units]
        dur, dem, exp = t1 - t0, morae[i], morae[i] / rate
        ratio = dur / max(1e-6, exp)
        # the honest misfit metric: onsets actually observed inside the span vs the
        # morae the line demands.  (ratio above is ~1 by construction, so it cannot
        # detect a misplaced line on its own.)
        n_on = int(((ot >= t0) & (ot < t1)).sum())
        on_per_mora = n_on / max(1e-6, dem)
        conf = 0.35 + 0.15 * (snapped[i] and snapped[i + 1]) + 0.10 * (0.6 <= ratio <= 1.6)
        conf = round(min(0.6, max(0.15, conf)), 2)
        words = []
        for tok, us, sp in zip(lu.tokens, lu.units, spans):
            d = {"w": tok, "start": sp[0][0], "end": sp[-1][1], "conf": conf}
            if len(sp) > 1:
                d["syl"] = [[a, b] for a, b in sp]
            words.append(d)
        out.append({"i": lu.i, "text": lu.text, "start": round(t0, 3), "end": round(t1, 3),
                    "words": words})
        rep.append({"i": lu.i, "start": round(t0, 3), "end": round(t1, 3), "dur": round(dur, 3),
                    "morae": round(dem, 1), "expected": round(exp, 3), "ratio": round(ratio, 2),
                    "units": lu.n_units, "conf": conf, "snapped": bool(snapped[i]),
                    "onsets": n_on, "on_per_mora": round(on_per_mora, 2)})
    flags = [r for r in rep if r["dur"] < min_line or r["dur"] > max_line
             or not (0.5 <= r["ratio"] <= 2.0) or not (0.4 <= r["on_per_mora"] <= 3.5)]
    return out, rep, flags, {"rate_morae_per_s": round(rate, 3), "active_s": round(sum(b - a for a, b, _ in segs), 2)}


def decode_mono(path, cache) -> np.ndarray:
    """Any audio file as 44.1 kHz mono float32, through the same static ffmpeg the mux path
    uses (so a decode of a stem is directly comparable with the decode of the mix).  Cached."""
    if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        cmd = [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(path),
               "-ac", "1", "-ar", str(SR), "-f", "f32le", str(cache)]
        subprocess.run(cmd, check=True)
    return np.fromfile(cache, dtype="<f4").astype(np.float32)


def stem_offset(x, ref, search=OFFSET_SEARCH_S, grid=OFFSET_GRID_MS) -> tuple[float, float]:
    """Constant delay of `x` against `ref`, in seconds (positive = `x` is the later one), plus
    the envelope correlation that was reached at it.

    The whole timing contract rests on analysis time == mux time (common.OFFSET_MS), and the
    Demucs stem honours it because Lane B decoded the very container the engine muxes.  A
    re-isolation made outside this repo does not: re-encoding through MP3, or a resample on the
    way out, leaves a constant delay of tens of milliseconds -- a whole syllable of error in a
    six-onset-per-second line, and invisible to everything downstream.

    Measured on the log-RMS envelope and not the waveform, because two different isolation
    models share almost no waveform (measured ncc ~0.02 against each other) while their loudness
    envelope is unmistakably the same performance (ncc ~0.98).  Only frames where `x` is actually
    singing are scored, so the delay is carried by the voice and not by a drum hit that survived
    in one of the two.
    """
    fps = 1000.0 / grid                                  # envelope rate of the search grid
    ex = np.log10(frame_rms(x, sr=SR, fps=fps) + 1e-4)
    er = np.log10(frame_rms(ref, sr=SR, fps=fps) + 1e-4)
    live = ex > np.percentile(ex, 40)                    # the sung majority of the frames
    n = min(len(ex), len(er))
    live[: int(20.0 * fps)] = False                      # never let the intro carry the delay
    idx = np.nonzero(live[:n])[0]
    x0 = er[idx] - er[idx].mean()
    nx = np.linalg.norm(x0) + 1e-9
    best, bs, bc = -np.inf, 0, 0.0
    for s in range(-int(search * fps), int(search * fps) + 1):
        j = idx + s
        ok = (j >= 0) & (j < n)
        if ok.sum() < 1000:
            continue
        y = ex[j[ok]]
        y = y - y.mean()
        c = float((x0[ok] * y).sum() / (nx * (np.linalg.norm(y) + 1e-9)))
        if c > best:
            best, bs, bc = c, s, c
    return bs / fps, bc


def slide_to(x, offset: float, n: int) -> np.ndarray:
    """Move `x` back onto `ref`'s timeline (undoing a measured `offset`) and cut/pad it to `n`
    samples.  The padding is the stretch the isolation dropped at the end of the file; the DP
    only ever reads onsets inside the vocal gate, so a silent tail costs nothing."""
    k = int(round(offset * SR))
    y = x[k:] if k >= 0 else np.concatenate([np.zeros(k, dtype=np.float32), x])
    if y.size < n:
        y = np.concatenate([y, np.zeros(n - y.size, dtype=np.float32)])
    return y[:n].astype(np.float32)


def load_vocal_stem(which: str = "stem") -> tuple[np.ndarray | None, float]:
    """The voice to work from, on the mix's own timeline: (samples, offset measured to get here).

      'stem'   the Demucs output Lane B produced (analysis/stems/.../vocals.wav); decoded from
               the same container as the mix, so it needs nothing.
      'vocal'  audio/vocals.mp3 -- an isolation made outside this repo.  Decoded, then measured
               against the mix and slid, because an outside tool does not keep mux time.
    """
    if which == "stem":
        p = STEMS / "vocals.wav"
        if not p.exists():
            return None, 0.0
        import soundfile as sf
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == SR, f"stem is {sr} Hz, expected {SR}"
        return y.mean(1).astype(np.float32), 0.0
    if which == "vocal":
        if not common.VOCALS.exists():
            return None, 0.0
        return decode_mono(common.VOCALS, common.WORK / "vocal44100.f32"), 0.0
    raise ValueError("no such vocal source: %r" % which)


def vocal_source(source: str, mix: np.ndarray) -> tuple[np.ndarray | None, float]:
    """-> (the voice aligned to `mix`'s length and clock, the offset that was applied to it).

    A foreign isolation is never measured against the mix: the mix's envelope is the drums and
    the synths for most of this song, and correlating a vocal stem to it wanders by hundreds of
    milliseconds (measured: +29/+111/-97/+134 ms over consecutive 10 s windows).  It is measured
    against the Demucs stem, whose own delay to the mix is zero by construction -- Lane B decoded
    the same container with the same decoder -- and which shares the singing, if none of the
    waveform, with any other isolation of the same take (envelope corr 0.93-0.99 everywhere).
    """
    if source == "mix":
        return None, 0.0
    stem, _ = load_vocal_stem(source)
    if stem is None:
        where = STEMS / "vocals.wav" if source == "stem" else common.VOCALS
        raise SystemExit("no %s vocal stem at %s -- run Lane B first" % (source, where))
    off, corr = 0.0, 1.0
    if source == "vocal":
        ref, _ = load_vocal_stem("stem")
        if ref is None:
            raise SystemExit("cannot place %s on the mix clock without the Demucs stem at %s"
                             % (common.VOCALS.name, STEMS / "vocals.wav"))
        off, corr = stem_offset(stem, ref)
        print("  %s sits %+6.1f ms late against the Demucs stem (envelope corr %.3f) -> slid back "
              "onto mux time; it is %.2f s shorter than the mix, so the tail is zero-padded"
              % (common.VOCALS.name, 1000 * off, corr, (mix.size - stem.size) / SR))
    else:
        assert stem.size == mix.size, "stem and mix differ in length: %d vs %d" % (stem.size, mix.size)
    y = slide_to(stem, off, mix.size) if off else stem[: mix.size]
    quiet = float(np.sqrt((y[: int(58 * SR)] ** 2).mean()))
    loud = float(np.sqrt((y[int(60 * SR): int(560 * SR)] ** 2).mean()))
    print("  vocal source %-5s %6.2f s  rms %.4f  singing rms %.4f  leak in the 58 s intro "
          "%.5f (= %5.1f dB below the singing)"
          % (source, y.size / SR, float(np.sqrt((y ** 2).mean())), loud, quiet,
             20.0 * np.log10((quiet + 1e-9) / (loud + 1e-9))))
    return y, off


def compute(source: str = "mix", target_notes: int | None = None) -> dict:
    """All L2 features.  `source`: 'mix' (Tier A proxy), 'stem' (Tier B, Demucs) or
    'vocal' (Tier B on the isolation in audio/vocals.mp3)."""
    x = read_mono()
    nf = n_frames(len(x))
    stem, offset = vocal_source(source, x)

    S_full = stft_mag(x)
    mel = mel_matrix()
    voice = stem if stem is not None else x
    Sv = stft_mag(voice)
    pres, strength, flat = vocal_features(Sv, mel)

    # gate: on the stem an absolute gate works (upstream's approach); on the mix the
    # harmonics of pads and synths never fall below it, so gate relative to a local median
    if stem is None:
        loc = median_filter(pres, size=301, mode="nearest")
        rel = pres / (loc + 1e-9)
        gate_env = smooth_env(np.clip(np.log10(rel + 1e-9) / 0.4 + 0.5, 0, 1))
        g_on, g_off = 0.56, 0.52          # ~ +3.0 / +1.0 dB above the local median
    else:
        gate_env = norm01(smooth_env(pres))
        g_on, g_off = 0.18, 0.10

    env = {
        "rms": norm01(smooth_env(frame_rms(x))),
        "low": norm01(smooth_env(band_rms(x, *BAND_LOW))),
        "mid": norm01(smooth_env(band_rms(x, *BAND_MID))),
        "high": norm01(smooth_env(band_rms(x, *BAND_HIGH))),
        "vocal": norm01(smooth_env(pres)),
    }
    env["vocal_raw"] = smooth_env(pres)          # before normalisation, for absolute gates
    runs = phrase_runs(gate_env, on=g_on, off=g_off)

    # note onsets: peaks of the vocal-source flux, but only where the voice is actually
    # present -- the flux alone fires on the noise floor (see vocal_features) -- and crowded
    # down to the note budget the text implies (see onset_peaks).
    strength = np.where(env["vocal"] > ONSET_FLOOR, strength, 0.0).astype(np.float32)
    idx, h = onset_peaks(strength, target=target_notes)
    onsets = np.stack([idx / FPS, h / (h.max() + 1e-9)])         # (2, n)

    return {
        "nf": nf, "source": source, "offset": offset, "env": env, "presence": pres,
        "strength": strength, "flat": flat, "runs": runs, "onsets": onsets, "S_full": S_full,
    }


def fit_beats(env, fps=FPS, bpm_lo=60.0, bpm_hi=200.0) -> dict:
    """Constant-tempo fit: autocorrelation for the period, grid search for the phase.

    Provisional (v0): the real beat/downbeat/section pass is Phase 2, on the stems.
    """
    x = env - env.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-12
    lo = int(fps * 60.0 / bpm_hi)
    hi = int(fps * 60.0 / bpm_lo)
    k = int(np.argmax(ac[lo:hi]) + lo)
    # parabolic refinement around the peak
    if 0 < k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    period = k / fps
    bpm = 60.0 / period
    best, phase = -1.0, 0.0
    for off in np.arange(0, k, 1.0):
        gi = np.round(np.arange(off, len(env), k)).astype(int)
        s = float(env[gi].sum())
        if s > best:
            best, phase = s, float(off / fps)
    beats = np.arange(phase, len(env) / fps, period)
    return {"bpm": float(bpm), "period": float(period), "phase": float(phase),
            "beats": beats.tolist(), "ac_peak": float(ac[int(round(k))])}


def phrase_runs(env, fps=FPS, on=0.18, off=0.10, min_run=0.15, merge=0.15):
    """Hysteresis gate -> [(start_s, end_s, mean, peak)], gaps < merge joined."""
    runs, i, n = [], 0, len(env)
    while i < n:
        if env[i] <= on:
            i += 1
            continue
        j = i
        while j + 1 < n and env[j + 1] > off:
            j += 1
        runs.append([i, j])
        i = j + 1
    merged: list[list[int]] = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) / fps < merge:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    out = []
    for a, b in merged:
        if (b - a + 1) / fps < min_run:
            continue
        seg = env[a: b + 1]
        out.append((a / fps, (b + 1) / fps, float(seg.mean()), float(seg.max())))
    return out


# --------------------------------------------------------------------------- plots
def plots(feat, rep, src, zooms=((0, 12), (60, 72), (150, 162), (287, 299), (430, 442), (560, 572))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(feat["nf"]) / FPS
    env, runs, ot = feat["env"], feat["runs"], feat["onsets"]
    lines_t = [(r["start"], r["end"], r["i"]) for r in rep]

    def panel(ax, t0, t1):
        m = (t >= t0) & (t <= t1)
        ax[0].plot(t[m], env["vocal"][m], lw=0.8, color="tab:purple", label="vocal presence")
        ax[0].plot(t[m], env["vocal_raw"][m] / (env["vocal_raw"].max() + 1e-9), lw=0.6,
                   color="tab:pink", alpha=0.8, label="raw harmonic 200-4k")
        for (a, b, mean, peak) in runs:
            if b > t0 and a < t1:
                ax[0].axvspan(max(a, t0), min(b, t1), color="tab:purple", alpha=0.12)
        oi = (ot[0] >= t0) & (ot[0] <= t1)
        ax[0].vlines(ot[0][oi], 0, ot[1][oi], color="k", lw=0.5, alpha=0.5)
        ax[0].set_ylabel("vocal / onsets")
        for name, c in (("rms", "k"), ("low", "tab:red"), ("mid", "tab:green"), ("high", "tab:blue")):
            ax[1].plot(t[m], env[name][m], lw=0.7, color=c, label=name)
        ax[1].legend(loc="upper right", fontsize=7, ncol=5)
        ax[1].set_ylabel("mix bands")
        for (a, b, i) in lines_t:
            for axx in (ax[0], ax[2]):
                axx.axvline(a, color="tab:orange", lw=0.7, alpha=0.85)
                axx.axvline(b, color="tab:orange", lw=0.4, alpha=0.4)
            if a >= t0 and a <= t1:
                ax[2].text(a, 0.55, str(i), fontsize=6, color="tab:orange", rotation=90)
                ax[0].text(a, 1.01, str(i), fontsize=6, color="tab:orange", rotation=90)
        ax[2].set_ylim(0, 1)
        ax[2].set_ylabel("v0 lines")
        ax[2].set_xlim(t0, t1)

    fig, ax = plt.subplots(3, 1, figsize=(28, 10), sharex=False,
                           gridspec_kw=dict(height_ratios=[2, 1.6, 0.7]))
    panel(ax, 0, t[-1])
    for a_ in ax:
        a_.set_xlim(0, t[-1])
        a_.grid(alpha=0.2)
    fig.suptitle("L2 %s: vocal proxy, phrase runs (%d), onsets (%d), v0 line spans" % (src, len(runs), ot.shape[1]))
    fig.tight_layout()
    fig.savefig(common.QA / ("l2_overview_%s.png" % src), dpi=70)
    plt.close(fig)

    for (t0, t1) in zooms:
        fig, ax = plt.subplots(3, 1, figsize=(20, 8), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1.6, 0.7]))
        panel(ax, t0, t1)
        for a_ in ax:
            a_.set_xticks(np.arange(np.ceil(t0), t1, 0.5))
        fig.suptitle("L2 %s zoom %.0f-%.0f s" % (src, t0, t1))
        fig.tight_layout()
        fig.savefig(common.QA / ("l2_zoom_%s_%03d.png" % (src, int(t0))), dpi=80)
        plt.close(fig)
    return len(zooms) + 1


# ---------------------------------------------------------------------------- main
def report(rep, flags, info, feat, beats, src) -> str:
    n_on = int(feat["onsets"].shape[1])
    onsets_info = (n_on, n_on / max(1e-6, info["active_s"]), sum(r["units"] for r in rep))
    o = ["L2 feature + v0 allocation report  (source: %s)" % src,
         "  frames            %d  (%.2f s at %d fps)" % (feat["nf"], feat["nf"] / FPS, FPS),
         "  stem delay        %+7.1f ms taken out to put this source on the mux clock"
         % (1000 * feat.get("offset", 0.0)),
         "  phrase runs       %d   active %.1f s   median %.2f s" % (
             len(feat["runs"]), info["active_s"],
             float(np.median([b - a for a, b, _, _ in feat["runs"]]))),
         "  note onsets       %d   (%.2f/s over the active span; %d units need spans)" % onsets_info,
         "  singing rate      %.2f morae/s (global fit)" % info["rate_morae_per_s"],
         "  tempo (provisional) %.2f BPM  period %.4f s  phase %.3f s  ac %.3f"
         % (beats["bpm"], beats["period"], beats["phase"], beats["ac_peak"]),
         "  confidence        %d lines >= 0.5, %d at 0.35" % (
             sum(1 for r in rep if r["conf"] >= 0.5), sum(1 for r in rep if r["conf"] < 0.4)),
         "",
         "  i   start     end    dur  morae  expect  ratio  units  conf  snap  onsets on/mora",
         ]
    for r in rep:
        o.append("  %3d %8.3f %8.3f %6.2f %6.1f %7.2f %6.2f %5d  %4.2f  %-3s  %5d %6.2f"
                 % (r["i"], r["start"], r["end"], r["dur"], r["morae"], r["expected"],
                    r["ratio"], r["units"], r["conf"], "yes" if r["snapped"] else "",
                    r["onsets"], r["on_per_mora"]))
    o.append("")
    o.append("  FLAGGED (%d): duration <0.25 s / >12 s or capacity ratio outside 0.5..2.0" % len(flags))
    for r in flags:
        o.append("     L%02d  %.2f s  ratio %.2f  morae %.1f" % (r["i"], r["dur"], r["ratio"], r["morae"]))
    return "\n".join(o)


def main(argv: list[str]) -> int:
    src = "vocal" if "--vocal" in argv else ("stem" if "--stems" in argv else "mix")
    lines, unresolved, ja, zh = lyric_units.parse()
    morae = [lu.morae()[1] for lu in lines]
    # The text supplies the note budget: one mora is one note in this repertoire, so the
    # detector may keep NOTE_BUDGET times the morae -- enough surplus that the alignment can
    # drop its weakest candidates instead of being forced to sing on them.
    feat = compute(src, target_notes=int(round(NOTE_BUDGET * sum(morae))))
    beats = fit_beats(feat["env"]["rms"])

    np.savez_compressed(
        common.WORK / ("features_%s.npz" % src),
        nf=feat["nf"], runs=np.array(feat["runs"], dtype=np.float32),
        onsets=feat["onsets"].astype(np.float32), strength=feat["strength"],
        flat=feat["flat"], beats=np.array(beats["beats"], dtype=np.float64),
        bpm=beats["bpm"], period=beats["period"], phase=beats["phase"],
        offset=np.float64(feat.get("offset", 0.0)),
        **{k: v for k, v in feat["env"].items()})

    out, rep, flags, info = allocate(lines, morae, feat["runs"], feat["onsets"])
    txt = report(rep, flags, info, feat, beats, src)
    (common.QA / ("l2_report_%s.txt" % src)).write_text(txt + "\n", encoding="utf-8")
    print(txt[:4000])

    if "--alloc" in argv:
        doc = {
            "lines": out,
            "extras": [],
            "notes": ("v0 approximate lyric timing (%s features).  Order-constrained "
                      "proportional allocation of %d lines over %d phrase runs (%.1f s of "
                      "activity), boundaries snapped into nearby silences, unit bounds "
                      "snapped to note onsets within 80 ms, unit weights = mora counts from "
                      "script orthography (Japanese kanji 1..3, prior 2; Chinese 1 per glyph; "
                      "kana 1 with digraph/long-vowel rules).  Expect ~+-0.3-1.0 s at line "
                      "starts; superseded by the stem-based pass (analysis/analyze.py --stems)."
                      % (src, len(out), len(feat["runs"]), info["active_s"])),
            "source": "analysis/analyze.py --alloc",
            "method": {"features": src, "runs": len(feat["runs"]), "onsets": int(feat["onsets"].shape[1]),
                       "rate_morae_per_s": info["rate_morae_per_s"], "tempo_bpm": round(beats["bpm"], 3)},
        }
        (common.DATA / "lyrics.approx.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("\nwrote %s" % (common.DATA / "lyrics.approx.json"))
        n = len(out)
        print("first lines: " + " | ".join("%d %.2f-%.2f" % (d["i"], d["start"], d["end"]) for d in out[:5]))
        print("last  lines: " + " | ".join("%d %.2f-%.2f" % (d["i"], d["start"], d["end"]) for d in out[-5:]))

    if "--no-plots" not in argv:
        print("plots: %d written to %s" % (plots(feat, rep, src), common.QA))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
