"""
Speech-to-text service using Groq's hosted Whisper Large V3.

Handles the Groq audio upload size limit transparently: files under the
limit are transcribed directly; larger files are split into sequential
chunks with FFmpeg (see `src.audio_utils`), transcribed independently, and
combined into one transcript in the original chronological order.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import groq

from src.audio_utils import needs_audio_chunking, split_audio_into_chunks

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails. Message is safe to show to users."""


class TranscriptionServiceBase(ABC):
    """Interface any speech-to-text backend should implement."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe the audio file at `audio_path` and return clean text."""
        raise NotImplementedError


class GroqTranscriptionService(TranscriptionServiceBase):
    """Transcribes audio using Groq's Whisper Large V3 API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        groq_audio_limit_bytes: int,
        client: groq.Groq | None = None,
    ) -> None:
        self._model = model
        self._audio_limit_bytes = groq_audio_limit_bytes
        # Allow a pre-built/mock client to be injected for testing so tests
        # never need a real API key or network access.
        self._client = client or groq.Groq(api_key=api_key)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe the audio file at `audio_path`, transparently chunking
        it first if it exceeds Groq's upload size limit.

        Raises TranscriptionError (never a raw SDK/OS exception) on any
        failure. Never logs the API key or full transcript content.
        """
        if not os.path.exists(audio_path):
            raise TranscriptionError("The audio file could not be read, or is empty.")

        size_bytes = os.path.getsize(audio_path)
        if size_bytes == 0:
            raise TranscriptionError("The audio file could not be read, or is empty.")

        logger.info("Transcription started (model=%s, size=%d bytes)", self._model, size_bytes)

        if needs_audio_chunking(size_bytes, self._audio_limit_bytes):
            transcript = self._transcribe_in_chunks(audio_path, size_bytes)
        else:
            transcript = self._transcribe_single_file(audio_path)

        logger.info("Transcription completed (%d characters)", len(transcript))

        if not transcript.strip():
            raise TranscriptionError("No usable speech was detected in this recording.")

        return transcript

    def _transcribe_in_chunks(self, audio_path: str, size_bytes: int) -> str:
        logger.info("Audio exceeds Groq upload limit; splitting into chunks")
        with split_audio_into_chunks(audio_path, size_bytes, self._audio_limit_bytes) as chunk_paths:
            transcripts = []
            for i, chunk_path in enumerate(chunk_paths, start=1):
                logger.info("Transcribing chunk %d/%d", i, len(chunk_paths))
                transcripts.append(self._transcribe_single_file(chunk_path))
            # Chunks are yielded in chronological order by split_audio_into_chunks
            # (sorted filenames), so joining preserves the original sequence.
            return " ".join(t.strip() for t in transcripts if t.strip())

    def _transcribe_single_file(self, audio_path: str) -> str:
        try:
            with open(audio_path, "rb") as audio_file:
                response = self._client.audio.transcriptions.create(
                    model=self._model,
                    file=(os.path.basename(audio_path), audio_file.read()),
                    response_format="text",
                )
        except groq.AuthenticationError as exc:
            logger.error("Groq authentication failed during transcription.")
            raise TranscriptionError(
                "Groq authentication failed. Please check your API key."
            ) from exc
        except groq.RateLimitError as exc:
            logger.error("Groq rate limit hit during transcription.")
            raise TranscriptionError(
                "The Groq API rate limit was reached. Please try again shortly."
            ) from exc
        except groq.APITimeoutError as exc:
            logger.error("Groq request timed out during transcription.")
            raise TranscriptionError(
                "The transcription request timed out. Please try again."
            ) from exc
        except groq.APIConnectionError as exc:
            logger.error("Network error while calling the Groq transcription API.")
            raise TranscriptionError(
                "Could not reach the Groq API. Please check your network connection."
            ) from exc
        except groq.BadRequestError as exc:
            logger.error("Groq rejected the transcription request as invalid.")
            raise TranscriptionError(
                "Groq rejected the audio file. Please check the file and try again."
            ) from exc
        except groq.APIStatusError as exc:
            logger.error("Groq API returned an error status: %s", exc.status_code)
            raise TranscriptionError(
                "Transcription failed. Please check the audio file and API configuration."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net
            logger.error("Unexpected error during transcription: %s", type(exc).__name__)
            raise TranscriptionError("Transcription failed due to an unexpected error.") from exc

        # response_format="text" returns a plain string; other formats (the
        # default "json") return an object with a `.text` attribute. Handle
        # both so this stays robust to a future response_format change.
        if isinstance(response, str):
            return response
        return getattr(response, "text", "") or ""
