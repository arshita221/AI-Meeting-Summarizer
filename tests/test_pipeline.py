"""
Tests covering config, schema parsing, transcript chunking, audio
validation/chunking (using real FFmpeg on a generated test tone — no
network involved), and the transcription/summarization services with the
Groq client mocked. No real API key or network access is required.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import groq
import pytest

from src.audio_utils import (
    FileValidationError,
    get_audio_duration_seconds,
    needs_audio_chunking,
    split_audio_into_chunks,
    validate_audio_upload,
)
from src.config import Config, ConfigError, load_config
from src.schemas import ActionItem, MeetingSummary, SummaryParsingError, parse_meeting_summary
from src.summarization import GroqSummarizationService, SummarizationError
from src.text_utils import chunk_transcript, clean_transcript, needs_chunking
from src.transcription import GroqTranscriptionService, TranscriptionError


def _fake_auth_response() -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/x")
    return httpx.Response(401, request=request, json={"error": {"message": "bad key"}})


def _make_test_audio(path: str, duration_seconds: int = 6) -> None:
    """Generate a short synthetic WAV tone with real FFmpeg (no network)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_seconds}",
            path,
        ],
        capture_output=True, check=True,
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class TestConfig:
    def test_valid_config_passes(self):
        config = Config(
            groq_api_key="fake-key",
            whisper_model="whisper-large-v3",
            llm_model="llama-3.3-70b-versatile",
            max_upload_mb=200,
            groq_audio_limit_mb=25,
            max_chars_per_chunk=12000,
        )
        config.validate()

    def test_missing_api_key_raises(self):
        config = Config(
            groq_api_key="",
            whisper_model="m",
            llm_model="m",
            max_upload_mb=200,
            groq_audio_limit_mb=25,
            max_chars_per_chunk=12000,
        )
        with pytest.raises(ConfigError, match="GROQ_API_KEY"):
            config.validate()

    def test_load_config_defaults(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        empty_env = tmp_path / ".env.empty"
        empty_env.write_text("")
        config = load_config(env_file=str(empty_env))
        assert config.whisper_model == "whisper-large-v3"
        assert config.llm_model == "llama-3.3-70b-versatile"
        assert config.groq_audio_limit_mb == 25

    def test_invalid_int_env_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_UPLOAD_MB", "not-a-number")
        empty_env = tmp_path / ".env.empty"
        empty_env.write_text("")
        with pytest.raises(ConfigError, match="MAX_UPLOAD_MB"):
            load_config(env_file=str(empty_env))
        monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)


# --------------------------------------------------------------------------
# Schema parsing (safe JSON parsing of LLM output)
# --------------------------------------------------------------------------


class TestSchemaParsing:
    def test_parses_well_formed_response(self):
        raw = """{"summary": "Discussed budget.", "discussion_points": ["Budget review"],
        "decisions": ["Approved increase"], "action_items": [
        {"task": "Send doc", "owner": "Sam", "deadline": "Friday", "priority": "high"}],
        "follow_up_items": ["Confirm vendor pricing"]}"""
        result = parse_meeting_summary(raw)
        assert result.summary == "Discussed budget."
        assert result.action_items[0].owner == "Sam"
        assert result.action_items[0].priority == "High"

    def test_missing_owner_defaults_to_not_specified(self):
        raw = '{"summary": "S", "action_items": [{"task": "Do X"}]}'
        result = parse_meeting_summary(raw)
        assert result.action_items[0].owner == "Not specified"
        assert result.action_items[0].deadline == "Not specified"

    def test_action_item_without_task_is_dropped(self):
        raw = '{"summary": "S", "action_items": [{"owner": "Sam"}, {"task": "Real task"}]}'
        result = parse_meeting_summary(raw)
        assert len(result.action_items) == 1
        assert result.action_items[0].task == "Real task"

    def test_missing_summary_raises(self):
        with pytest.raises(SummaryParsingError):
            parse_meeting_summary('{"discussion_points": ["a"]}')

    def test_invalid_json_raises(self):
        with pytest.raises(SummaryParsingError):
            parse_meeting_summary("not json at all")

    def test_non_object_json_raises(self):
        with pytest.raises(SummaryParsingError):
            parse_meeting_summary("[1, 2, 3]")

    def test_to_markdown_and_to_json_round_trip(self):
        summary = MeetingSummary(
            summary="S",
            action_items=[ActionItem(task="T", owner="O", deadline="D", priority="High")],
        )
        md = summary.to_markdown("Test Meeting")
        assert "# Test Meeting" in md
        assert "T" in md and "O" in md
        rebuilt = parse_meeting_summary(summary.to_json())
        assert rebuilt.summary == summary.summary


# --------------------------------------------------------------------------
# Transcript text utilities
# --------------------------------------------------------------------------


class TestTextUtils:
    def test_clean_transcript_collapses_whitespace(self):
        assert clean_transcript("Hello   world.\n\n\n\nBye.") == "Hello world.\n\nBye."

    def test_short_transcript_not_chunked(self):
        text = "Short transcript."
        assert not needs_chunking(text, max_chars=1000)
        assert chunk_transcript(text, max_chars=1000) == [text]

    def test_long_transcript_is_chunked_at_sentence_boundaries(self):
        sentence = "We discussed the roadmap today. "
        text = sentence * 50
        chunks = chunk_transcript(text, max_chars=200)
        assert len(chunks) > 1
        rejoined = " ".join(chunks)
        assert sentence.strip() in rejoined


# --------------------------------------------------------------------------
# Audio validation
# --------------------------------------------------------------------------


class TestAudioValidation:
    def test_valid_upload_passes(self):
        validate_audio_upload("meeting.mp3", size_bytes=1024, max_upload_bytes=10_000_000)

    def test_invalid_extension_rejected(self):
        with pytest.raises(FileValidationError, match="Unsupported audio format"):
            validate_audio_upload("meeting.txt", size_bytes=1024, max_upload_bytes=10_000_000)

    def test_empty_file_rejected(self):
        with pytest.raises(FileValidationError, match="empty or corrupt"):
            validate_audio_upload("meeting.mp3", size_bytes=0, max_upload_bytes=10_000_000)

    def test_oversized_file_rejected(self):
        with pytest.raises(FileValidationError, match="too large"):
            validate_audio_upload("meeting.mp3", size_bytes=50_000, max_upload_bytes=10_000)


# --------------------------------------------------------------------------
# Real FFmpeg-based audio chunking (no network — verifies the actual
# subprocess pipeline works end-to-end on a generated test tone)
# --------------------------------------------------------------------------


class TestAudioChunking:
    def test_duration_detection_on_real_file(self, tmp_path):
        audio_path = tmp_path / "tone.wav"
        _make_test_audio(str(audio_path), duration_seconds=4)
        duration = get_audio_duration_seconds(str(audio_path))
        assert 3.5 <= duration <= 4.5

    def test_needs_audio_chunking(self):
        assert needs_audio_chunking(size_bytes=30_000_000, limit_bytes=25_000_000)
        assert not needs_audio_chunking(size_bytes=1_000_000, limit_bytes=25_000_000)

    def test_split_audio_into_chunks_on_real_file(self, tmp_path):
        audio_path = tmp_path / "tone.wav"
        # 70 seconds is comfortably longer than the service's 30-second
        # minimum segment length, so an aggressively small size limit
        # (forcing the 30s floor) reliably yields multiple chunks.
        _make_test_audio(str(audio_path), duration_seconds=70)
        size_bytes = audio_path.stat().st_size

        with split_audio_into_chunks(str(audio_path), size_bytes, limit_bytes=1024) as chunks:
            assert len(chunks) >= 2
            for chunk_path in chunks:
                assert chunk_path.endswith(".wav")


# --------------------------------------------------------------------------
# Transcription service (Groq client mocked)
# --------------------------------------------------------------------------


class TestTranscriptionService:
    def test_successful_transcription(self, tmp_path):
        audio_path = tmp_path / "meeting.mp3"
        audio_path.write_bytes(b"fake-audio-bytes")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "Hello, this is a test transcript."
        service = GroqTranscriptionService(
            api_key="fake-key", model="whisper-large-v3",
            groq_audio_limit_bytes=25_000_000, client=mock_client,
        )

        result = service.transcribe(str(audio_path))
        assert result == "Hello, this is a test transcript."

    def test_missing_file_raises(self):
        service = GroqTranscriptionService(
            api_key="fake-key", model="m", groq_audio_limit_bytes=25_000_000, client=MagicMock()
        )
        with pytest.raises(TranscriptionError):
            service.transcribe("/nonexistent/audio.mp3")

    def test_empty_transcription_result_raises(self, tmp_path):
        audio_path = tmp_path / "meeting.mp3"
        audio_path.write_bytes(b"fake-audio-bytes")
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "   "
        service = GroqTranscriptionService(
            api_key="fake-key", model="m", groq_audio_limit_bytes=25_000_000, client=mock_client
        )
        with pytest.raises(TranscriptionError, match="No usable speech"):
            service.transcribe(str(audio_path))

    def test_authentication_error_is_wrapped(self, tmp_path):
        audio_path = tmp_path / "meeting.mp3"
        audio_path.write_bytes(b"fake-audio-bytes")
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = groq.AuthenticationError(
            "bad key", response=_fake_auth_response(), body=None
        )
        service = GroqTranscriptionService(
            api_key="fake-key", model="m", groq_audio_limit_bytes=25_000_000, client=mock_client
        )
        with pytest.raises(TranscriptionError, match="authentication failed"):
            service.transcribe(str(audio_path))

    def test_large_file_is_chunked_and_combined(self, tmp_path):
        audio_path = tmp_path / "meeting.wav"
        _make_test_audio(str(audio_path), duration_seconds=70)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = [
            "First chunk text.", "Second chunk text.", "Third chunk text.",
            "Fourth chunk text.", "Fifth chunk text.",
        ]
        service = GroqTranscriptionService(
            api_key="fake-key", model="m", groq_audio_limit_bytes=1024, client=mock_client
        )

        result = service.transcribe(str(audio_path))
        assert "First chunk text." in result
        assert mock_client.audio.transcriptions.create.call_count >= 2


# --------------------------------------------------------------------------
# Summarization service (Groq client mocked)
# --------------------------------------------------------------------------


def _chat_response(json_content: str):
    message = SimpleNamespace(content=json_content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class TestSummarizationService:
    def test_single_pass_summarization(self):
        raw_json = '{"summary": "Team discussed Q3 planning.", "action_items": []}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _chat_response(raw_json)
        service = GroqSummarizationService(api_key="fake-key", model="m", client=mock_client)

        result = service.summarize("A short transcript.", max_chars_per_chunk=10_000)
        assert result.summary == "Team discussed Q3 planning."

    def test_empty_transcript_raises(self):
        service = GroqSummarizationService(api_key="fake-key", model="m", client=MagicMock())
        with pytest.raises(SummarizationError, match="No usable speech"):
            service.summarize("   ", max_chars_per_chunk=10_000)

    def test_malformed_json_retries_then_raises(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _chat_response("not valid json")
        service = GroqSummarizationService(api_key="fake-key", model="m", client=mock_client)

        with pytest.raises(SummarizationError):
            service.summarize("Some transcript.", max_chars_per_chunk=10_000)
        assert mock_client.chat.completions.create.call_count == 2

    def test_recovers_after_one_bad_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _chat_response("not valid json"),
            _chat_response('{"summary": "Recovered summary."}'),
        ]
        service = GroqSummarizationService(api_key="fake-key", model="m", client=mock_client)

        result = service.summarize("Some transcript.", max_chars_per_chunk=10_000)
        assert result.summary == "Recovered summary."

    def test_authentication_error_is_wrapped(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq.AuthenticationError(
            "bad key", response=_fake_auth_response(), body=None
        )
        service = GroqSummarizationService(api_key="fake-key", model="m", client=mock_client)
        with pytest.raises(SummarizationError, match="authentication failed"):
            service.summarize("Some transcript text.", max_chars_per_chunk=10_000)

    def test_long_transcript_chunked_and_combined(self):
        chunk_json = '{"summary": "Partial summary."}'
        final_json = '{"summary": "Combined final summary."}'

        def fake_create(**kwargs):
            system_msg = kwargs["messages"][0]["content"]
            if system_msg.startswith("You are an AI meeting assistant. You will be given"):
                return _chat_response(final_json)
            return _chat_response(chunk_json)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create
        service = GroqSummarizationService(api_key="fake-key", model="m", client=mock_client)

        sentence = "We discussed the roadmap in detail today. "
        long_transcript = sentence * 50
        result = service.summarize(long_transcript, max_chars_per_chunk=200)

        assert result.summary == "Combined final summary."
        assert mock_client.chat.completions.create.call_count > 1
