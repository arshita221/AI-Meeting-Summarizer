"""
Audio file validation, temp-file handling, and FFmpeg-based chunking.

Uploaded audio is never written to permanent storage — only to a temporary
file/directory for the duration of processing, which is always cleaned up
afterward, success or failure.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from src.config import EXPECTED_MIME_TYPES, SUPPORTED_AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation. Message is user-safe."""


class AudioProcessingError(Exception):
    """Raised when FFmpeg-based inspection/splitting fails. User-safe message."""


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_audio_upload(
    filename: str,
    size_bytes: int,
    max_upload_bytes: int,
    mime_type: str | None = None,
) -> None:
    """Validate an uploaded audio file before any processing happens."""
    if not filename or not filename.strip():
        raise FileValidationError("No file was uploaded.")

    extension = get_extension(filename)
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise FileValidationError(
            "Unsupported audio format. Please upload MP3, WAV, M4A, MP4, WEBM, or OGG."
        )

    if size_bytes <= 0:
        raise FileValidationError(
            "The uploaded file appears to be empty or corrupt. Please try a different file."
        )

    if size_bytes > max_upload_bytes:
        max_mb = max_upload_bytes / (1024 * 1024)
        raise FileValidationError(
            f"File is too large. The maximum upload size is {max_mb:.0f} MB."
        )

    if mime_type:
        expected = EXPECTED_MIME_TYPES.get(extension)
        if expected and mime_type not in expected:
            logger.warning(
                "MIME type mismatch for upload: extension=%s declared_mime=%s",
                extension,
                mime_type,
            )


@contextlib.contextmanager
def temp_audio_file(file_bytes: bytes, filename: str) -> Iterator[str]:
    """Write `file_bytes` to a temp file with the original extension; always cleaned up."""
    extension = get_extension(filename) or ".tmp"
    fd, path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not remove temporary audio file: %s", path)


def _run_ffmpeg(args: list[str]) -> None:
    """Run an ffmpeg/ffprobe command, raising AudioProcessingError on failure."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=600, check=False
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "FFmpeg is not installed or not available on the system PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError("Audio processing timed out.") from exc

    if result.returncode != 0:
        # Never surface raw ffmpeg stderr to the end user; log it instead.
        logger.error("FFmpeg command failed (%s): %s", args[0], result.stderr[-2000:])
        raise AudioProcessingError("Audio processing failed while preparing the file.")


def get_audio_duration_seconds(audio_path: str) -> float:
    """Return the duration of an audio file in seconds, via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "FFprobe is not installed or not available on the system PATH."
        ) from exc

    if result.returncode != 0 or not result.stdout.strip():
        raise AudioProcessingError("Could not read the audio file's duration.")

    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioProcessingError("Could not read the audio file's duration.") from exc


def needs_audio_chunking(size_bytes: int, limit_bytes: int) -> bool:
    return size_bytes > limit_bytes


@contextlib.contextmanager
def split_audio_into_chunks(
    audio_path: str, size_bytes: int, limit_bytes: int
) -> Iterator[list[str]]:
    """
    Split an audio file into sequential chunks that each stay under
    `limit_bytes`, using FFmpeg's segment muxer with stream copy (no
    re-encoding, so this is fast and lossless).

    Splitting is duration-based: assuming roughly constant bitrate, the
    target chunk duration is derived from the ratio of the size limit to
    the file's average bytes-per-second. This is a practical approximation
    appropriate for an MVP, not a byte-exact guarantee for variable
    bitrate files (a small safety margin is applied to compensate).

    Yields a list of chunk file paths in chronological order. The temp
    directory holding them is always removed on exit.
    """
    duration = get_audio_duration_seconds(audio_path)
    if duration <= 0:
        raise AudioProcessingError("The audio file has no measurable duration.")

    bytes_per_second = size_bytes / duration
    # 15% safety margin: stream-copy segment boundaries aren't byte-exact,
    # so aim comfortably under the limit rather than right at it.
    safe_limit = limit_bytes * 0.85
    segment_seconds = max(int(safe_limit / bytes_per_second), 30)

    extension = get_extension(audio_path) or ".mp3"
    tmp_dir = tempfile.mkdtemp(prefix="meeting_audio_chunks_")
    pattern = os.path.join(tmp_dir, f"chunk_%03d{extension}")

    try:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                "-f", "segment",
                "-segment_time", str(segment_seconds),
                "-c", "copy",
                "-reset_timestamps", "1",
                pattern,
            ]
        )
        chunk_paths = sorted(
            str(p) for p in Path(tmp_dir).glob(f"chunk_*{extension}")
        )
        if not chunk_paths:
            raise AudioProcessingError("Audio splitting produced no output chunks.")
        logger.info(
            "Split audio into %d chunks (~%ds each)", len(chunk_paths), segment_seconds
        )
        yield chunk_paths
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
