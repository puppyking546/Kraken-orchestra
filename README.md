# 🦑 Kraken Orchestra

Make a song sound like it's being played by Kraken X60 motors, like CTRE Phoenix 6 Orchestra, but simulated on your computer. No robot needed.

## Files

| File | What it does |
|---|---|
| `kraken_orchestra.py` | MIDI in, motor-sound WAV out |
| `kraken_orchestra_audio.py` | mp3/wav in. Splits vocals, bass, drums and harmony across separate motors. Needs `kraken_orchestra.py` in the same folder |
| `kraken_orchestra_colab.ipynb` | Runs the mp3 version on free Google Colab (no installing) |
| `example_scale.mid` | A tiny MIDI file to test with |
| `requirements.txt` | Python packages |

## Quick start (MIDI)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy mido
python kraken_orchestra.py example_scale.mid --motors 6 --room
```

Output: `example_scale_kraken.wav`. Options: `--motors N`, `--room`, `--min-hz`, `--max-hz`, `--seed`.

## mp3 version

```bash
pip install -r requirements.txt
sudo apt install ffmpeg        # mp3 decoding (Linux)
python kraken_orchestra_audio.py song.mp3 --start 30 --duration 45 --room
```

Try a short clip first. Stem separation is slow on a normal CPU. The easiest way is Google Colab: open `kraken_orchestra_colab.ipynb` at colab.research.google.com (File → Upload notebook, or the GitHub tab). In its step 2, set `GITHUB_REPO` to `YOUR_USERNAME/kraken-orchestra`. Your repo has to be public for that to work.

## How the motors are assigned

- vocals: 1 motor
- bass: 1 motor
- guitar / piano / synth: 5 motors by default (`--other-motors N`)
- drums: 1 percussion motor

Each motor plays one note at a time at a fixed volume, like the real Orchestra. Notes outside the motors' range are shifted by octaves.

## Notes

- The sound is a model of a motor, not a recording of one. Tweak `render_motor()` in `kraken_orchestra.py` to taste.
- To run on a real robot, Phoenix Tuner X converts MIDI to `.chrp`.
- Only use music you have the right to use, especially if you share outputs.
