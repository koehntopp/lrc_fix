# lrc_fix

Sync lyrics to audio in FLAC files, locally.

Reads the `LYRICS` tag from a FLAC file — plain text or an existing
(possibly out-of-sync) LRC — and rewrites it with timestamps generated
straight from the audio. No cloud APIs, no LLM in the timing path: every
timestamp is grounded in an actual detected word and a real onset spike in
the waveform.

## How it works

1. **Isolate vocals** — [demucs](https://github.com/facebookresearch/demucs)
   separates the track into vocal/instrumental stems so instrumental
   sections never get fed to the speech recognizer (the #1 cause of
   hallucinated lyrics).
2. **Transcribe** — [whisperx](https://github.com/m-bain/whisperX)
   transcribes and force-aligns the isolated vocals, producing word-level
   timestamps.
3. **Align** — the correct lyric lines (from the tag) are matched against
   the ASR transcript with deterministic word-sequence matching
   ([`difflib`](https://docs.python.org/3/library/difflib.html)). Only runs
   of 2+ consecutive matched words are trusted, so a single coincidental
   word match can't lock in a bogus timestamp.
4. **Snap to onset** — each matched timestamp is pulled back to the nearest
   real vocal onset detected via
   [librosa](https://librosa.org/doc/latest/generated/librosa.onset.onset_detect.html),
   landing on the actual start of the word instead of somewhere in the
   middle of it.
5. **Write back** — the FLAC's `LYRICS` tag is rewritten with the new LRC
   content. The original tag value is preserved once under
   `LYRICS_ORIGINAL`, so every subsequent run realigns from the pristine
   original rather than compounding on the tool's own prior output.

## Requirements

- macOS/Linux with Python and [`uv`](https://docs.astral.sh/uv/) installed.
- No manual dependency install needed — `lrc_fix.py` carries its
  dependencies as inline [PEP 723](https://peps.python.org/pep-0723/)
  metadata; `uv` resolves and caches them on first run.
- CPU works fine (slower); pass `--device cuda` if you have a GPU.

## Install

```bash
curl -O https://raw.githubusercontent.com/<you>/lrc_fix/main/lrc_fix.py
chmod +x lrc_fix.py
```

Or just clone the repo.

## Usage

```bash
# single file
./lrc_fix.py song.flac

# every .flac under a directory, recursively
./lrc_fix.py ~/Music/some_album/

# preview without writing the tag
./lrc_fix.py song.flac --dry-run

# force language instead of relying on auto-detect
./lrc_fix.py song.flac --language en
```

A file is skipped if it has no `LYRICS` tag, or if the tag is empty after
stripping timestamps/header lines.

### Options

| Flag | Default | Description |
|---|---|---|
| `--whisper-model` | `medium` | whisperx/faster-whisper model size (`tiny`…`large-v3`) |
| `--language` | auto-detect | Force ASR language (ISO 639-1, e.g. `en`, `de`) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--compute-type` | `int8` | faster-whisper compute type |
| `--dry-run` | off | Print the resulting LRC, don't write the tag |
| `--no-isolate-vocals` | off | Skip demucs separation, transcribe the full mix |
| `--no-snap-onsets` | off | Skip onset detection/snapping |
| `--dump-words` | off | Print the raw ASR word/timestamp transcript (debugging) |

## Notes

- First run downloads whisperx/demucs model weights from Hugging Face Hub
  and caches them (`~/.cache/torch`, `~/.cache/huggingface`) — later runs
  are fast to start.
- `--whisper-model large*` is rarely worth it here: since the correct
  lyrics text is already known, the alignment step only needs *usable*
  ASR output to match against, not maximally accurate transcription.
- Standard LRC header lines (`[ar:...]`, `[ti:...]`, `[al:...]`) in the tag
  are preserved as-is and never timestamped.

## License

MIT (or whatever you prefer — update this section).
