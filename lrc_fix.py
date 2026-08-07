#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "mutagen",
#     "whisperx",
#     "demucs",
#     "librosa",
#     "torch>=2.4,<2.6",
#     "torchaudio>=2.4,<2.6",
#     "numpy<2",
# ]
# ///
"""lrc_fix.py

Read the LYRICS tag from FLAC files (plain text or already-LRC-timestamped).
First isolate vocals with demucs (source separation) so instrumental
sections can't get mistaken for speech, then use whisperx to get real
word-level timestamps from the isolated vocals, then align the correct
lyric lines against that ASR transcript with deterministic word sequence
matching (difflib) to produce a synced LRC. The LYRICS tag is rewritten in
place, parsed fresh from its current content each run - so manual edits to
the tag (fixing a misheard word, adding a line that repeats but was only
written once) take effect on the next run. No backup is kept; if a run goes
wrong, delete the tag and re-fetch lyrics (e.g. from lrclib.net) instead.

Alignment is intentionally not LLM-based: local models are unreliable at
precise positional/numeric reasoning over a long transcript (they can lose
track, guess, or lock onto the wrong occurrence of a repeated phrase).
Sequence alignment grounds every timestamp in an actual matched ASR word.

Usage:
    python lrc_fix.py song.flac
    python lrc_fix.py ~/Music/some_album/
    python lrc_fix.py ~/Music/some_album/ --whisper-model medium --dry-run

Requires: mutagen, whisperx, demucs
"""
from __future__ import annotations

import argparse
import difflib
import logging
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

# Silence harmless version-mismatch / deprecation noise from whisperx's deps
# (torch.load weights_only default, pyannote/torch checkpoint version checks).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)

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


def ensure_id_tags(id_tags: list[str], flac: FLAC) -> list[str]:
    """Fill in missing [ar:]/[ti:]/[al:] header lines from the FLAC's own tags,
    and stamp [re:CREATOR_TAG] to identify this tool as the file's creator.

    ar/ti/al lines already present are left untouched; only keys missing
    entirely get synthesized from the file's ARTIST/TITLE/ALBUM vorbis
    comments (when present). Any pre-existing [re:] line is replaced (it
    records which tool last touched the file, not user data). Any other
    pre-existing id tag lines (e.g. [by:...]) are kept, in their original
    order, after the standard ones.
    """
    by_key = {}
    others = []
    for line in id_tags:
        m = _ID_TAG_KEY_RE.match(line)
        key = m.group(1).lower() if m else ""
        if key in dict(_STANDARD_ID_TAGS):
            by_key[key] = line
        elif key != "re":
            others.append(line)

    result = []
    for key, vorbis_key in _STANDARD_ID_TAGS:
        if key in by_key:
            result.append(by_key[key])
        else:
            values = flac.get(vorbis_key)
            if values and values[0]:
                result.append(f"[{key}:{values[0]}]")
    result.append(f"[re:{CREATOR_TAG}]")
    result.extend(others)
    return result


def separate_vocals(audio_path: Path, device: str, work_dir: Path) -> Path:
    """Run demucs source separation and return the path to the vocals-only stem.

    Isolating vocals before transcription keeps instrumental sections from
    being mistaken for speech, which is whisper's main hallucination trigger.
    """
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-d", device,
        "-o", str(work_dir),
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"demucs failed:\n{proc.stderr}")

    vocals_path = next(work_dir.rglob("vocals.wav"), None)
    if vocals_path is None:
        raise RuntimeError(f"demucs did not produce a vocals.wav under {work_dir}")
    return vocals_path


def transcribe_words(audio_path: Path, whisper_model: str, device: str,
                      compute_type: str, language: str | None) -> list[dict]:
    import whisperx

    # condition_on_previous_text=False avoids whisper's known failure mode of
    # feeding a hallucinated guess (e.g. from an instrumental intro) back in
    # as context and cascading it into further hallucinated text.
    asr_options = {"condition_on_previous_text": False}
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type,
                                 language=language, asr_options=asr_options)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=language, print_progress=True)

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(result["segments"], align_model, metadata, audio, device,
                             print_progress=True)

    words = []
    for segment in result["segments"]:
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
            print("   isolating vocals (demucs)...")
            audio_path = separate_vocals(path, args.device, Path(tmp))

        print(f"   transcribing ({args.whisper_model}, {args.device})...")
        words = transcribe_words(audio_path, args.whisper_model, args.device,
                                  args.compute_type, args.language)

        onsets = []
        if not args.no_snap_onsets:
            print("   detecting vocal onsets...")
            onsets = detect_onsets(audio_path)

    if not words:
        print("   skip: whisperx produced no word timestamps")
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
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
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

    if args.path.is_dir() and not args.all:
        kept = []
        skipped = 0
        for f in files:
            try:
                raw = read_lyrics_tag(FLAC(f))
            except Exception as exc:
                print(f"error reading {f}: {exc}", file=sys.stderr)
                continue
            if raw and is_lrc(raw):
                skipped += 1
            else:
                kept.append(f)
        if skipped:
            print(f"skipping {skipped} file(s) already LRC-timestamped (use --all to include them)")
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
