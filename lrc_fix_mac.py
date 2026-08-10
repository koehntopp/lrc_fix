#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "mutagen",
#     "mlx-whisper",
#     "demucs",
#     "librosa",
#     "torch>=2.4,<2.6",
#     "torchaudio>=2.4,<2.6",
#     "numpy<2",
# ]
# ///
"""lrc_fix_mac.py - Apple Silicon variant of lrc_fix.py

EXPERIMENTAL test copy. Same LRC-sync pipeline as lrc_fix.py, retargeted at
Apple Silicon acceleration where it actually works:

- Transcription uses mlx-whisper (Apple's MLX framework, Metal/ANE-native)
  instead of whisperx/faster-whisper. mlx-whisper also produces word-level
  timestamps directly (its own DTW-based alignment), so the separate
  whisperx + wav2vec2 forced-alignment step is gone entirely - one model
  call instead of two, no CTranslate2 dependency, and this is the stage
  that actually gets meaningfully faster here (it was the dominant compute
  cost, and CTranslate2 has no MPS support at all to begin with).
- demucs vocal separation stays on `--device cpu` by default. "mps" is
  available but NOT the default: the default demucs model (htdemucs) has a
  conv1d layer with >65536 output channels, a hard Metal dimension limit
  that PYTORCH_ENABLE_MPS_FALLBACK=1 does not work around (confirmed by
  testing - it's a deliberate raise in the MPS kernel itself, not a missing
  op the generic fallback dispatch catches). "mps" only works here with a
  non-htdemucs model (e.g. `mdx_extra`) that doesn't hit that limit.

librosa onset detection remains NumPy/SciPy - no GPU path either way.

See lrc_fix.py's docstring/CLAUDE.md for the rest of the pipeline design
(vocal isolation to avoid hallucination, difflib word-sequence alignment
instead of an LLM, onset snapping, id-tag handling, etc.) - all unchanged
here.

Usage:
    python lrc_fix_mac.py song.flac
    python lrc_fix_mac.py ~/Music/some_album/
    python lrc_fix_mac.py ~/Music/some_album/ --whisper-model medium --dry-run

Requires: mutagen, mlx-whisper, demucs (Apple Silicon Mac only)
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

# Silence harmless deprecation noise from torch/demucs internals.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mutagen.flac import FLAC

LRC_TIMESTAMP_RE = re.compile(r"^\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]\s*")
LRC_ID_TAG_RE = re.compile(r"^\[[a-zA-Z]{2,10}:[^\]]*\]$")


def find_flac_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".flac" else []
    return sorted(path.rglob("*.flac"))


def read_lyrics_tag(flac: FLAC) -> str | None:
    values = flac.get("LYRICS")
    if not values:
        return None
    return "\n".join(values)


def is_lrc(raw: str) -> bool:
    """True if any line already carries an [mm:ss.xx]-style timestamp."""
    return any(LRC_TIMESTAMP_RE.match(line.strip()) for line in raw.splitlines())


_INSTRUMENTAL_RE = re.compile(r"^[\[(]?\s*instrumental\s*[\])]?$", re.IGNORECASE)


def is_instrumental(lyric_lines: list[str]) -> bool:
    """True if the tag is just a placeholder like 'Instrumental'/'[Instrumental]'."""
    return len(lyric_lines) == 1 and bool(_INSTRUMENTAL_RE.match(lyric_lines[0]))


def strip_lyric_lines(raw: str) -> tuple[list[str], list[str]]:
    """Split raw LYRICS content into (id_tags, lyric_lines).

    id_tags are standard LRC header lines like [ar:...]/[ti:...]/[al:...] and
    are passed through unchanged, never timestamped. Any existing per-line
    timestamps on lyric lines are stripped so they get realigned from audio.
    """
    id_tags = []
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip any (possibly stacked, from a prior buggy run) leading
        # timestamp(s) before classifying the line
        while True:
            stripped = LRC_TIMESTAMP_RE.sub("", line, count=1).strip()
            if stripped == line:
                break
            line = stripped
        if not line:
            continue
        if LRC_ID_TAG_RE.match(line):
            id_tags.append(line)
        else:
            lines.append(line)
    return id_tags, lines


_STANDARD_ID_TAGS = [("ar", "ARTIST"), ("ti", "TITLE"), ("al", "ALBUM")]
_ID_TAG_KEY_RE = re.compile(r"^\[([a-zA-Z]+):")

# LRC spec's [re:] tag identifies the player/editor that created the file.
# Always overwritten on write - it names the tool, not user-owned data.
CREATOR_TAG = "https://github.com/koehntopp/lrc_fix"


def format_length_tag(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds - minutes * 60)
    return f"[length:{minutes:02d}:{secs:02d}]"


def ensure_id_tags(id_tags: list[str], flac: FLAC) -> list[str]:
    """Fill in missing [ar:]/[ti:]/[al:]/[length:] header lines from the FLAC's
    own tags/audio duration, and stamp [re:CREATOR_TAG] to identify this tool
    as the file's creator.

    ar/ti/al lines already present are left untouched; only keys missing
    entirely get synthesized from the file's ARTIST/TITLE/ALBUM vorbis
    comments (when present). [length:] is handled the same way but sourced
    from flac.info.length (there's no vorbis comment for it) rather than
    left for some other tool to add later. Any pre-existing [re:] line is
    replaced (it records which tool last touched the file, not user data).
    Any other pre-existing id tag lines (e.g. [by:...]) are kept, in their
    original order, after the standard ones.
    """
    by_key = {}
    others = []
    has_length = False
    for line in id_tags:
        m = _ID_TAG_KEY_RE.match(line)
        key = m.group(1).lower() if m else ""
        if key in dict(_STANDARD_ID_TAGS):
            by_key[key] = line
        elif key != "re":
            others.append(line)
            if key == "length":
                has_length = True

    result = []
    for key, vorbis_key in _STANDARD_ID_TAGS:
        if key in by_key:
            result.append(by_key[key])
        else:
            values = flac.get(vorbis_key)
            if values and values[0]:
                result.append(f"[{key}:{values[0]}]")
    result.append(f"[re:{CREATOR_TAG}]")
    if not has_length and getattr(flac.info, "length", None):
        result.append(format_length_tag(flac.info.length))
    result.extend(others)
    return result


def separate_vocals(audio_path: Path, device: str, work_dir: Path) -> Path:
    """Run demucs source separation and return the path to the vocals-only stem.

    Isolating vocals before transcription keeps instrumental sections from
    being mistaken for speech, which is whisper's main hallucination trigger.
    demucs is a plain PyTorch model, so `device` runs it on GPU when set to
    "mps"/"cuda" - but see the "cpu" default note below.

    htdemucs (the default demucs model) has a conv1d layer with >65536
    output channels, which is a hard Metal dimension limit, not merely an
    unimplemented op - PYTORCH_ENABLE_MPS_FALLBACK=1 does NOT help here (the
    MPS conv kernel raises deliberately once past the channel limit, so the
    generic fallback dispatch never triggers). That's why `device` defaults
    to "cpu" in main() for this script despite otherwise targeting Apple
    Silicon: "mps" reliably crashes on htdemucs. It's kept as an option here
    for anyone using a non-htdemucs model (e.g. `mdx_extra`) that doesn't
    hit this limit. PYTORCH_ENABLE_MPS_FALLBACK=1 is still set below as a
    harmless safety net for other, genuinely-unimplemented ops such a model
    might hit.
    """
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-d", device,
        "-o", str(work_dir),
        str(audio_path),
    ]
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"demucs failed:\n{proc.stderr}")

    vocals_path = next(work_dir.rglob("vocals.wav"), None)
    if vocals_path is None:
        raise RuntimeError(f"demucs did not produce a vocals.wav under {work_dir}")
    return vocals_path


def _mlx_model_repo(whisper_model: str) -> str:
    """Map a plain size name to its mlx-community HF repo, e.g. "medium" ->
    "mlx-community/whisper-medium-mlx". Pass a full repo id or local path
    directly (must contain "/") to bypass this and use it as-is.
    """
    if "/" in whisper_model:
        return whisper_model
    return f"mlx-community/whisper-{whisper_model}-mlx"


def transcribe_words(audio_path: Path, whisper_model: str, device: str,
                      language: str | None) -> list[dict]:
    """Transcribe with mlx-whisper, which runs on Apple's GPU/ANE via MLX and
    returns word-level timestamps directly (its own DTW-based alignment) -
    no separate forced-alignment pass needed. `device` is unused here (MLX
    always targets the available Metal GPU); it only matters for demucs.
    """
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=_mlx_model_repo(whisper_model),
        language=language,
        word_timestamps=True,
        # Avoids whisper's known failure mode of feeding a hallucinated
        # guess (e.g. from an instrumental intro) back in as context and
        # cascading it into further hallucinated text.
        condition_on_previous_text=False,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            start = w.get("start")
            if start is None:
                continue
            words.append({"w": w["word"], "t": round(start, 2)})
    return words


_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def match_line_times(lyric_lines: list[str], words: list[dict]) -> list[float | None]:
    """Find each lyric line's start time from an actual matched ASR word.

    Tokenizes both the ASR transcript and the lyric lines to plain lowercase
    words, then runs difflib's sequence matcher between them. A lone matched
    word is easy to hit by chance (e.g. a hallucinated phrase over an
    instrumental intro sharing a common word like "you" with some lyric
    line), so we only trust matching runs of 2+ consecutive words; lines that
    only ever get a single-word match fall back to that. Lines with no match
    at all come back as None (left for the caller to interpolate).
    """
    asr_tokens: list[str] = []
    asr_times: list[float] = []
    for w in words:
        toks = _tokenize(w["w"])
        if not toks:
            continue
        asr_tokens.append(toks[0])
        asr_times.append(w["t"])

    lyric_tokens: list[str] = []
    lyric_owner: list[int] = []
    for i, line in enumerate(lyric_lines):
        for tok in _tokenize(line):
            lyric_tokens.append(tok)
            lyric_owner.append(i)

    strong: list[float | None] = [None] * len(lyric_lines)
    weak: list[float | None] = [None] * len(lyric_lines)
    if asr_tokens and lyric_tokens:
        matcher = difflib.SequenceMatcher(a=asr_tokens, b=lyric_tokens, autojunk=False)
        for block in matcher.get_matching_blocks():
            target = strong if block.size >= 2 else weak
            for k in range(block.size):
                a_idx = block.a + k
                b_idx = block.b + k
                line_idx = lyric_owner[b_idx]
                t = asr_times[a_idx]
                if target[line_idx] is None or t < target[line_idx]:
                    target[line_idx] = t

    return [s if s is not None else weak[i] for i, s in enumerate(strong)]


def detect_onsets(audio_path: Path) -> list[float]:
    """Detect vocal onset times (seconds) in an audio file via librosa.

    backtrack=True walks each detected onset back to the nearest preceding
    local energy minimum, i.e. the actual start of the attack rather than
    its peak - what we want a lyric line's timestamp to land on.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    return sorted(float(t) for t in onsets)


def snap_to_onsets(times: list[float | None], onsets: list[float],
                    back_window: float = 0.7, fwd_window: float = 0.2) -> None:
    """Snap each known (non-None) time to the nearest real onset just at/before it.

    Modifies `times` in place. Falls back to the nearest onset slightly after
    the estimate if none is found within the backward window, and leaves the
    estimate untouched if no onset is close enough either way.
    """
    if not onsets:
        return
    for i, t in enumerate(times):
        if t is None:
            continue
        candidates = [o for o in onsets if t - back_window <= o <= t + fwd_window]
        if not candidates:
            continue
        before = [o for o in candidates if o <= t]
        times[i] = max(before) if before else min(candidates)


def finalize_times(times: list[float | None], onsets: list[float] | None = None) -> list[float]:
    times = list(times)
    if onsets:
        snap_to_onsets(times, onsets)
    _fill_and_clamp(times)
    return times  # type: ignore[return-value]


def _fill_and_clamp(times: list[float | None]) -> None:
    n = len(times)
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        for i in range(n):
            times[i] = 0.0
        return

    for i in range(known[0]):
        times[i] = times[known[0]]
    for a, b in zip(known, known[1:]):
        if b - a > 1:
            t0, t1 = times[a], times[b]
            for i in range(a + 1, b):
                frac = (i - a) / (b - a)
                times[i] = t0 + frac * (t1 - t0)
    for i in range(known[-1] + 1, n):
        times[i] = times[known[-1]]

    last = 0.0
    for i, t in enumerate(times):
        if t < last:
            t = last
        times[i] = t
        last = t


def format_lrc_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def build_lrc(id_tags: list[str], lyric_lines: list[str], times: list[float]) -> str:
    timed = (f"{format_lrc_timestamp(t)}{line}" for line, t in zip(lyric_lines, times))
    return "\n".join([*id_tags, *timed])


def write_lyrics_tag(flac: FLAC, new_lrc: str, dry_run: bool) -> None:
    print(new_lrc)
    if dry_run:
        return
    flac["LYRICS"] = new_lrc
    flac.save()


def process_file(path: Path, args: argparse.Namespace, index: int, total: int) -> None:
    print(f"== [{index}/{total}] {path}")
    flac = FLAC(path)
    raw = read_lyrics_tag(flac)
    if not raw:
        print("   skip: no LYRICS tag")
        return

    id_tags, lyric_lines = strip_lyric_lines(raw)
    if not lyric_lines:
        print("   skip: LYRICS tag empty after parsing")
        return
    if is_instrumental(lyric_lines):
        print("   skip: instrumental (no lyrics to align)")
        return
    id_tags = ensure_id_tags(id_tags, flac)

    with tempfile.TemporaryDirectory(prefix="lrc_fix_") as tmp:
        audio_path = path
        if not args.no_isolate_vocals:
            print(f"   isolating vocals (demucs, {args.device})...")
            audio_path = separate_vocals(path, args.device, Path(tmp))

        print(f"   transcribing ({args.whisper_model}, mlx)...")
        words = transcribe_words(audio_path, args.whisper_model, args.device,
                                  args.language)

        onsets = []
        if not args.no_snap_onsets:
            print("   detecting vocal onsets...")
            onsets = detect_onsets(audio_path)

    if not words:
        print("   skip: mlx-whisper produced no word timestamps")
        return

    if args.dump_words:
        for w in words:
            print(f"   {w['t']:8.2f}  {w['w']}")

    print(f"   aligning {len(lyric_lines)} lines...")
    raw_times = match_line_times(lyric_lines, words)
    times = finalize_times(raw_times, onsets)

    new_lrc = build_lrc(id_tags, lyric_lines, times)
    write_lyrics_tag(flac, new_lrc, args.dry_run)
    print(f"   {'would update' if args.dry_run else 'updated'} LYRICS tag")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="FLAC file or directory to process")
    parser.add_argument("--whisper-model", default="medium",
                         help="model size (tiny/base/small/medium/large-v3), or a "
                              "full mlx-community HF repo id / local path")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default="cpu",
                         help="device for demucs (cpu/mps/cuda); defaults to cpu because "
                              "the default demucs model (htdemucs) hits a hard MPS "
                              "channel-count limit and reliably crashes on mps - try mps "
                              "only with a non-htdemucs model. Transcription always runs "
                              "on MLX's Metal backend regardless of this flag")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all", action="store_true",
                         help="when path is a directory, also reprocess files whose "
                              "LYRICS tag is already LRC-timestamped")
    parser.add_argument("--no-isolate-vocals", action="store_true",
                         help="skip demucs vocal separation, transcribe the full mix")
    parser.add_argument("--no-snap-onsets", action="store_true",
                         help="don't snap line timestamps to detected vocal onsets")
    parser.add_argument("--dump-words", action="store_true",
                         help="print the raw ASR word/timestamp transcript for debugging")
    args = parser.parse_args()

    files = find_flac_files(args.path)
    if not files:
        print(f"no .flac files found under {args.path}", file=sys.stderr)
        return 1

    if args.path.is_dir():
        kept = []
        skipped_lrc = 0
        skipped_no_lyrics = 0
        for f in files:
            try:
                raw = read_lyrics_tag(FLAC(f))
            except Exception as exc:
                print(f"error reading {f}: {exc}", file=sys.stderr)
                continue
            if not raw:
                skipped_no_lyrics += 1
                continue
            if not args.all and is_lrc(raw):
                skipped_lrc += 1
                continue
            _, lyric_lines = strip_lyric_lines(raw)
            if not lyric_lines or is_instrumental(lyric_lines):
                skipped_no_lyrics += 1
                continue
            kept.append(f)
        if skipped_lrc:
            print(f"skipping {skipped_lrc} file(s) already LRC-timestamped (use --all to include them)")
        if skipped_no_lyrics:
            print(f"skipping {skipped_no_lyrics} file(s) with no usable lyrics")
        files = kept

    if not files:
        print("nothing to process")
        return 0

    print(f"{len(files)} file(s) to process")
    for i, path in enumerate(files, 1):
        try:
            process_file(path, args, i, len(files))
        except Exception as exc:
            print(f"   error: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
