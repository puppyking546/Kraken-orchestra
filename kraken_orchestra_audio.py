#!/usr/bin/env python3
"""
Kraken Orchestra - MP3 edition
------------------------------
Give it an mp3/wav. It splits the song into stems (vocals, bass, drums,
everything else), figures out the notes in each stem, and assigns each stem
its own simulated Kraken motors:

    vocals -> 1 motor   (melody line)
    bass   -> 1 motor   (bass line, folded up where motors can actually sing)
    other  -> N motors  (guitar/piano/synth: chords and harmony)
    drums  -> 1 motor   (kick/snare/hat as motor ticks and thumps)

Keep this file in the SAME FOLDER as kraken_orchestra.py (it reuses its
motor sound model).

Setup:
    pip install numpy scipy mido librosa soundfile torch demucs
    # optional but recommended for better chords/harmony:
    pip install basic-pitch
    # mp3 decoding needs ffmpeg:  sudo apt install ffmpeg   (ChromeOS Linux)

Usage:
    python kraken_orchestra_audio.py song.mp3
    python kraken_orchestra_audio.py song.mp3 --other-motors 6 --room
    python kraken_orchestra_audio.py song.mp3 --start 30 --duration 45   # test a clip first!

Stem separation is the slow part (several minutes on CPU for a full song;
Google Colab with a GPU is much faster). Stems are cached next to the output
so re-running with different settings is fast. Use --redo to force it.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import binary_closing

try:
    import librosa
except ImportError:
    sys.exit("Missing librosa. Run:  pip install librosa soundfile")

try:
    from kraken_orchestra import render_motor, add_room, SR
except ImportError:
    sys.exit("Put kraken_orchestra.py in the same folder as this file.")

AN_SR = 22050   # analysis sample rate
HOP = 512       # analysis hop (~23 ms)
HOP_S = HOP / AN_SR


# ----------------------------------------------------------------------------
# 1. Stem separation (Demucs)
# ----------------------------------------------------------------------------
def separate(path, start, duration):
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError:
        sys.exit("Missing demucs/torch. Run:  pip install torch demucs")

    y, _ = librosa.load(path, sr=44100, mono=False, offset=start,
                        duration=duration)
    if y.ndim == 1:
        y = np.stack([y, y])
    print(f"Loaded {y.shape[1] / 44100:.1f}s of audio. Separating stems "
          "(this is the slow part)...")

    model = get_model("htdemucs")
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wav = torch.from_numpy(y).float()
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        src = apply_model(model, wav[None], device=dev, split=True,
                          overlap=0.25, progress=True)[0]
    src = src * (ref.std() + 1e-8) + ref.mean()
    return {name: src[i].mean(0).cpu().numpy()
            for i, name in enumerate(model.sources)}


# ----------------------------------------------------------------------------
# 2. Transcription
# ----------------------------------------------------------------------------
def mono_notes(y, fmin, fmax):
    """Monophonic melody -> [(start, end, midi, amp)] using pyin."""
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=AN_SR,
                                 frame_length=2048, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n], rms[:n]
    thr = 0.08 * np.percentile(rms, 95)
    midi = np.where(voiced & ~np.isnan(f0), librosa.hz_to_midi(f0), np.nan)

    notes, seg, gap = [], [], 0

    def flush():
        if len(seg) * HOP_S >= 0.07:
            idx = [i for i, _ in seg]
            pitch = int(round(np.median([m for _, m in seg])))
            notes.append((idx[0] * HOP_S, (idx[-1] + 1) * HOP_S, pitch,
                          float(np.mean(rms[idx]))))
        seg.clear()

    for i in range(n):
        m = midi[i]
        if np.isnan(m) or rms[i] < thr:
            gap += 1
            if gap > 3 and seg:
                flush()
            continue
        if seg and abs(m - np.median([x[1] for x in seg[-8:]])) > 0.8:
            flush()
        gap = 0
        seg.append((i, m))
    flush()
    return notes


def poly_notes_cqt(y, max_poly):
    """Fallback polyphonic transcription from CQT peaks."""
    nb = 72  # C2..B7
    C = np.abs(librosa.cqt(y, sr=AN_SR, hop_length=HOP,
                           fmin=librosa.note_to_hz("C2"), n_bins=nb,
                           bins_per_octave=12))
    gmax = C.max() + 1e-9
    act = np.zeros_like(C)
    for t in range(C.shape[1]):
        col = C[:, t]
        pk = np.where((col[1:-1] > col[:-2]) & (col[1:-1] >= col[2:]))[0] + 1
        pk = pk[(col[pk] > 0.10 * gmax) & (col[pk] > 0.3 * col.max())]
        pk = pk[np.argsort(col[pk])[::-1][:max_poly]]
        act[pk, t] = col[pk] / gmax
    notes = []
    for k in range(nb):
        a = binary_closing(act[k] > 0, structure=np.ones(3))
        d = np.diff(np.concatenate([[0], a.astype(int), [0]]))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s >= 4:
                notes.append((s * HOP_S, e * HOP_S, 36 + k,
                              float(act[k, s:e].mean())))
    return notes


def poly_notes(y44, y22, max_poly):
    """Prefer basic-pitch if installed; otherwise CQT peak picking."""
    try:
        from basic_pitch.inference import predict
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "other.wav")
            sf.write(p, y44, 44100)
            _, _, ev = predict(p)
        print("  (using basic-pitch for harmony)")
        return [(float(s), float(e), int(p), float(a))
                for s, e, p, a, *_ in ev]
    except Exception as ex:
        print(f"  (basic-pitch unavailable: using simpler CQT method)")
        return poly_notes_cqt(y22, max_poly)


def detect_drums(y):
    """Drum stem -> [(time, kind)] with kind in kick/snare/hat."""
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP))
    fr = librosa.fft_frequencies(sr=AN_SR, n_fft=1024)
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=AN_SR)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=AN_SR,
                                        hop_length=HOP)
    low, high = S[fr < 200].sum(0), S[fr > 3500].sum(0)
    hits = []
    for f in frames:
        r = low[f:f + 3].sum() / (high[f:f + 3].sum() + 1e-9)
        kind = "kick" if r > 1.5 else ("hat" if r < 0.35 else "snare")
        hits.append((f * HOP_S, kind))
    return hits


# ----------------------------------------------------------------------------
# 3. Voice allocation and drum motor
# ----------------------------------------------------------------------------
def fold(freq, lo, hi):
    while freq < lo:
        freq *= 2
    while freq > hi:
        freq /= 2
    return freq


def allocate(notes, n_motors, lo, hi):
    """Each motor plays one note at a time; louder notes win conflicts."""
    voices = [[] for _ in range(n_motors)]
    for s, e, m, a in sorted(notes):
        f = fold(440.0 * 2 ** ((m - 69) / 12), lo, hi)
        e = max(e, s + 0.05)
        free = [i for i, v in enumerate(voices) if not v or v[-1][1] <= s]
        if free:
            i = max(free, key=lambda k: voices[k][-1][1] if voices[k] else -1)
        else:
            i = min(range(n_motors), key=lambda k: voices[k][-1][3])
            if a < voices[i][-1][3] * 0.8:
                continue  # new note too quiet, drop it
            voices[i][-1][1] = s
        voices[i].append([s, e, f, a])
    return [[(s, e, f) for s, e, f, _ in v] for v in voices]


def render_drums(hits, total, rng):
    n = int(total * SR) + 1
    out = np.zeros(n)
    for t, kind in hits:
        a = int(t * SR)
        if kind == "kick":
            L = int(0.10 * SR); tt = np.arange(L) / SR
            f = 90 + 140 * np.exp(-tt * 40)
            ph = 2 * np.pi * np.cumsum(f) / SR
            burst = (0.6 * np.sign(np.sin(ph)) + 0.4 * np.sin(ph)) * np.exp(-tt / 0.045)
        elif kind == "snare":
            L = int(0.13 * SR); tt = np.arange(L) / SR
            ph = 2 * np.pi * 190 * tt
            burst = (0.8 * rng.standard_normal(L) * np.exp(-tt / 0.04)
                     + 0.5 * np.sign(np.sin(ph)) * np.exp(-tt / 0.03))
        else:
            L = int(0.04 * SR); tt = np.arange(L) / SR
            burst = 0.6 * rng.standard_normal(L) * np.exp(-tt / 0.009)
        L = min(L, n - a)
        if L > 0:
            out[a:a + L] += burst[:L]
    sos = signal.butter(2, [120, 6000], btype="band", fs=SR, output="sos")
    return np.tanh(1.5 * signal.sosfilt(sos, out))


# ----------------------------------------------------------------------------
# 4. Build the orchestra
# ----------------------------------------------------------------------------
def pan_gains(p):
    return np.cos(p * np.pi / 2), np.sin(p * np.pi / 2)


def build(stems, args):
    rng = np.random.default_rng(args.seed)
    total = len(next(iter(stems.values()))) / 44100 + 0.6
    n = int(total * SR) + 1
    mix = np.zeros((n, 2))
    mid = 0

    def add(sig, pan, gain):
        l, r = pan_gains(pan)
        m = min(len(sig), n)
        mix[:m, 0] += sig[:m] * l * gain
        mix[:m, 1] += sig[:m] * r * gain

    r22 = {k: librosa.resample(v, orig_sr=44100, target_sr=AN_SR)
           for k, v in stems.items()}

    if not args.no_vocals and "vocals" in stems:
        print("Transcribing vocals...")
        nt = mono_notes(r22["vocals"], 80, 1000)
        ev = allocate([(s, e, m, a) for s, e, m, a in nt], 1, 180, 1400)[0]
        print(f"  {len(ev)} vocal notes -> 1 motor")
        add(render_motor(ev, total, rng, mid), 0.5, 1.0); mid += 1

    if not args.no_bass and "bass" in stems:
        print("Transcribing bass...")
        nt = mono_notes(r22["bass"], 35, 400)
        ev = allocate(nt, 1, 80, 400)[0]
        print(f"  {len(ev)} bass notes -> 1 motor")
        add(render_motor(ev, total, rng, mid), 0.42, 0.8); mid += 1

    if args.other_motors > 0 and "other" in stems:
        print("Transcribing harmony (guitar/piano/synth)...")
        nt = poly_notes(stems["other"], r22["other"], args.other_motors + 2)
        voices = allocate(nt, args.other_motors, 150, 2200)
        k = args.other_motors
        for i, ev in enumerate(voices):
            pan = 0.5 if k == 1 else 0.1 + 0.8 * i / (k - 1)
            print(f"  harmony motor {i + 1}/{k}: {len(ev)} notes")
            add(render_motor(ev, total, rng, mid), pan, 0.55); mid += 1

    if not args.no_drums and "drums" in stems:
        print("Detecting drum hits...")
        hits = detect_drums(r22["drums"])
        print(f"  {len(hits)} hits -> 1 percussion motor")
        add(render_drums(hits, total, rng), 0.55, 0.7)

    if args.room:
        mix = add_room(mix, rng)
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.89
    return mix


def main():
    ap = argparse.ArgumentParser(description="MP3 -> Kraken motor orchestra")
    ap.add_argument("audio")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--other-motors", type=int, default=5,
                    help="motors for guitar/piano/synth harmony (default 5)")
    ap.add_argument("--no-vocals", action="store_true")
    ap.add_argument("--no-bass", action="store_true")
    ap.add_argument("--no-drums", action="store_true")
    ap.add_argument("--start", type=float, default=0.0, help="start second")
    ap.add_argument("--duration", type=float, default=None,
                    help="only process this many seconds (good for testing)")
    ap.add_argument("--room", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--redo", action="store_true", help="redo stem separation")
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.audio)[0] + "_kraken.wav"
    cache = out + ".stems.npz"

    if os.path.exists(cache) and not args.redo:
        print(f"Reusing cached stems: {cache}")
        stems = dict(np.load(cache))
    else:
        stems = separate(args.audio, args.start, args.duration)
        np.savez_compressed(cache, **stems)

    mix = build(stems, args)
    wavfile.write(out, SR, (mix * 32767).astype(np.int16))
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
