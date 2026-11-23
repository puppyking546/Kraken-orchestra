#!/usr/bin/env python3
"""
Kraken Orchestra Simulator
--------------------------
Give it a MIDI file, get back a WAV that sounds like a bank of Kraken X60
motors playing the song (like CTRE Phoenix 6 Orchestra, but no robot needed).

How the real thing works (and what this imitates):
  * Each motor is a single monophonic voice (one note at a time).
  * Pitch comes from driving the stator coils at the note frequency, so the
    "sound" is the motor housing buzzing, not a speaker.
  * There is no volume control: every note is the same loudness.
  * Notes the motors can't sing well (too low/too high) get folded by octaves.

Usage:
    pip install numpy scipy mido
    python kraken_orchestra.py song.mid
    python kraken_orchestra.py song.mid -o out.wav --motors 8 --seed 3

Options:
    --motors N     number of simulated motors/voices (default 6)
    --min-hz HZ    lowest playable pitch (default 150)
    --max-hz HZ    highest playable pitch (default 2200)
    --no-stereo    mono output (default spreads motors across a "robot")
    --room         add a little shop/room reverb
    --seed N       randomize motor personalities (default 1)

Need audio (mp3/wav) instead of MIDI? Convert it to MIDI first with a
transcription tool (e.g. Spotify's basic-pitch), then feed that in here.
To run on a real robot, Phoenix Tuner X can convert MIDI -> .chrp.
"""
import argparse
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import mido
except ImportError:
    sys.exit("Missing dependency. Run:  pip install numpy scipy mido")

SR = 44100


# ----------------------------------------------------------------------------
# MIDI -> list of notes
# ----------------------------------------------------------------------------
def read_notes(path):
    """Return a list of (start_s, end_s, midi_note), skipping drums."""
    mid = mido.MidiFile(path)
    t = 0.0
    active = {}  # (channel, note) -> start
    notes = []
    for msg in mid:  # iterating a MidiFile yields delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.channel == 9:
                continue
            active[(msg.channel, msg.note)] = t
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active:
                start = active.pop(key)
                notes.append((start, t, msg.note))
    for (ch, n), start in active.items():  # unterminated notes
        notes.append((start, t, n))
    notes.sort()
    return notes


def fold_to_range(freq, lo, hi):
    """Octave-shift a pitch into the motor's playable range."""
    while freq < lo:
        freq *= 2
    while freq > hi:
        freq /= 2
    return freq


# ----------------------------------------------------------------------------
# Voice allocation: each motor plays one note at a time
# ----------------------------------------------------------------------------
def allocate(notes, n_motors, lo, hi):
    voices = [[] for _ in range(n_motors)]  # each: list of [start, end, freq]
    for start, end, note in notes:
        freq = fold_to_range(440.0 * 2 ** ((note - 69) / 12), lo, hi)
        if end - start < 0.02:
            end = start + 0.02
        # prefer a motor that's free right now
        free = [i for i, v in enumerate(voices) if not v or v[-1][1] <= start]
        if free:
            # pick the one that's been idle the shortest (keeps lines coherent)
            i = max(free, key=lambda k: voices[k][-1][1] if voices[k] else -1)
        else:
            # steal the motor whose current note started earliest
            i = min(range(n_motors), key=lambda k: voices[k][-1][0])
            voices[i][-1][1] = start  # cut the old note short
        voices[i].append([start, end, freq])
    return voices


# ----------------------------------------------------------------------------
# Motor sound model
# ----------------------------------------------------------------------------
def render_motor(events, total, rng, motor_id):
    n = int(total * SR) + 1
    freq = np.zeros(n)
    gate = np.zeros(n)
    onsets = []
    for s, e, f in events:
        a, b = int(s * SR), min(int(e * SR), n)
        freq[a:b] = f
        gate[a:b] = 1.0
        onsets.append(a)

    if not onsets:
        return np.zeros(n)

    # Each motor has its own personality
    detune = 1 + rng.uniform(-0.004, 0.004)
    duty = rng.uniform(0.22, 0.38)
    res_hz = rng.uniform(1800, 3400)  # housing resonance
    flutter_hz = rng.uniform(5, 9)

    # Tiny pitch wobble (current-loop jitter), phase from integrated freq
    wobble = 1 + 0.0015 * np.sin(2 * np.pi * flutter_hz * np.arange(n) / SR)
    inst = freq * detune * wobble
    phase = np.cumsum(inst) / SR  # cycles
    frac = phase % 1.0

    # Drive waveform: narrow pulse (commutated phase current) + a little sine
    pulse = np.where(frac < duty, 1.0, -1.0) * 0.8
    pulse -= pulse.mean() * 0
    sine = np.sin(2 * np.pi * phase)
    wave = 0.65 * pulse + 0.35 * sine

    # PWM switching hiss (~20 kHz carrier aliasing/noise), only while driven
    hiss = rng.standard_normal(n)
    sos_h = signal.butter(2, [3000, 9000], btype="band", fs=SR, output="sos")
    hiss = signal.sosfilt(sos_h, hiss) * 0.06

    # Click when a note starts (rotor "ticks" as the field steps)
    click = np.zeros(n)
    for a in onsets:
        L = min(int(0.006 * SR), n - a)
        click[a:a + L] += (rng.standard_normal(L)
                           * np.exp(-np.arange(L) / (0.0012 * SR)) * 0.5)

    # Gate envelope: fast attack, short release (motors can't do dynamics)
    k = int(0.004 * SR)
    env = np.convolve(gate, np.ones(k) / k, mode="same")
    env = np.minimum(env, 1.0)

    raw = wave * env + hiss * env + click

    # Body: housing resonance (peaking bandpass) + gentle low-pass
    sos_res = signal.butter(2, [res_hz * 0.7, res_hz * 1.3], btype="band",
                            fs=SR, output="sos")
    resonant = signal.sosfilt(sos_res, raw)
    sos_lp = signal.butter(2, 6500, btype="low", fs=SR, output="sos")
    body = signal.sosfilt(sos_lp, raw + 1.4 * resonant)

    # Slight low-end "hum" of the rotor at the note fundamental via soft clip
    return np.tanh(1.6 * body)


def add_room(stereo, rng):
    """Small-room reverb from a synthetic decaying-noise impulse response."""
    L = int(0.35 * SR)
    t = np.arange(L) / SR
    out = []
    for ch in range(stereo.shape[1]):
        ir = rng.standard_normal(L) * np.exp(-t / 0.08)
        ir[0] = 1.0
        wet = signal.fftconvolve(stereo[:, ch], ir * 0.12)[: stereo.shape[0]]
        out.append(stereo[:, ch] + wet)
    return np.stack(out, axis=1)


def main():
    ap = argparse.ArgumentParser(description="Kraken motor orchestra simulator")
    ap.add_argument("midi")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--motors", type=int, default=6)
    ap.add_argument("--min-hz", type=float, default=150.0)
    ap.add_argument("--max-hz", type=float, default=2200.0)
    ap.add_argument("--no-stereo", action="store_true")
    ap.add_argument("--room", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out_path = args.output or args.midi.rsplit(".", 1)[0] + "_kraken.wav"
    rng = np.random.default_rng(args.seed)

    notes = read_notes(args.midi)
    if not notes:
        sys.exit("No notes found in that MIDI file.")
    total = max(e for _, e, _ in notes) + 0.6
    print(f"{len(notes)} notes, {total:.1f}s, {args.motors} motors")

    voices = allocate(notes, args.motors, args.min_hz, args.max_hz)

    n = int(total * SR) + 1
    mix = np.zeros((n, 2))
    for i, ev in enumerate(voices):
        print(f"  rendering motor {i + 1}/{args.motors} ({len(ev)} notes)")
        m = render_motor(ev, total, rng, i)
        pan = 0.5 if args.motors == 1 else i / (args.motors - 1)
        if args.no_stereo:
            pan = 0.5
        mix[:, 0] += m * np.cos(pan * np.pi / 2)
        mix[:, 1] += m * np.sin(pan * np.pi / 2)

    if args.room:
        mix = add_room(mix, rng)
    if args.no_stereo:
        mix = np.repeat(mix.mean(axis=1, keepdims=True), 2, axis=1)

    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.89
    wavfile.write(out_path, SR, (mix * 32767).astype(np.int16))
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
