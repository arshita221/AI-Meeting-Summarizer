"""
Centralized application configuration.

All environment-variable access happens here, and nowhere else in the
codebase, so model names, limits, and paths are never hard-coded in
multiple files, and missing/invalid configuration is caught early with a
clear, user-facing error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Formats accepted by the uploader and forwarded to Groq's Whisper API.
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg"}

EXPECTED_MIME_TYPES = {
    ".mp3": {"audio/mpeg", "audio/mp3"},
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave"},
    ".m4a": {"audio/m4a", "audio/x-m4a", "audio/mp4"},
    ".mp4": {"audio/mp4", "video/mp4"},
    ".webm": {"audio/webm", "video/webm"},
    ".ogg": {"audio/ogg", "video/ogg", "application/ogg"},
}


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of application configuration."""

    groq_api_key: str
    whisper_model: str
    llm_model: str
    max_upload_mb: int
    groq_audio_limit_mb: int
    max_chars_per_chunk: int

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def groq_audio_limit_bytes(self) -> int:
        return self.groq_audio_limit_mb * 1024 * 1024

    def validate(self) -> None:
        """Raise ConfigError if any setting is missing or nonsensical."""
        if not self.groq_api_key or not self.groq_api_key.strip():
            raise ConfigError(
                "Groq API key is not configured. Add GROQ_API_KEY to your .env file."
            )
        if not self.whisper_model.strip():
            raise ConfigError("GROQ_WHISPER_MODEL must not be empty.")
        if not self.llm_model.strip():
            raise ConfigError("GROQ_LLM_MODEL must not be empty.")
        if self.max_upload_mb <= 0:
            raise ConfigError("MAX_UPLOAD_MB must be a positive integer.")
        if self.groq_audio_limit_mb <= 0:
            raise ConfigError("GROQ_AUDIO_LIMIT_MB must be a positive integer.")
        if self.max_chars_per_chunk <= 0:
            raise ConfigError("MAX_CHARS_PER_CHUNK must be a positive integer.")


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw!r}") from exc


def load_config(env_file: str | Path | None = None) -> Config:
    """
    Load configuration from environment variables (and a `.env` file if
    present). Does NOT validate — call `.validate()` at the point where a
    user-facing error should be raised.
    """
    load_dotenv(dotenv_path=env_file, override=False)

    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        whisper_model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3").strip(),
        llm_model=os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile").strip(),
        max_upload_mb=_get_int_env("MAX_UPLOAD_MB", 200),
        # Groq's audio transcription endpoint currently caps request size at
        # 25MB (free tier); files above this are split into chunks first.
        groq_audio_limit_mb=_get_int_env("GROQ_AUDIO_LIMIT_MB", 25),
        max_chars_per_chunk=_get_int_env("MAX_CHARS_PER_CHUNK", 12000),
    )
