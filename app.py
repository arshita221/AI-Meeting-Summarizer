"""
AI Meeting Summarizer (Groq) - Streamlit application entry point.

This file is intentionally thin: it wires together the modules in `src/`
and renders the UI. Business logic (transcription, summarization,
validation) lives in `src/`, not here.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.audio_utils import FileValidationError, temp_audio_file, validate_audio_upload
from src.config import ConfigError, load_config
from src.schemas import MeetingSummary
from src.summarization import GroqSummarizationService, SummarizationError
from src.transcription import GroqTranscriptionService, TranscriptionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("meeting_summarizer.app")

st.set_page_config(page_title="AI Meeting Summarizer", page_icon="📝", layout="wide")


# --------------------------------------------------------------------------
# Cached resources / service construction
# --------------------------------------------------------------------------


@st.cache_resource
def get_config():
    return load_config()


def get_services(config):
    """Build fresh service instances using the current, validated config."""
    transcription = GroqTranscriptionService(
        api_key=config.groq_api_key,
        model=config.whisper_model,
        groq_audio_limit_bytes=config.groq_audio_limit_bytes,
    )
    summarization = GroqSummarizationService(
        api_key=config.groq_api_key, model=config.llm_model
    )
    return transcription, summarization


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def process_meeting(uploaded_file, config, status) -> dict | None:
    """Run the full pipeline: validate -> transcribe -> summarize."""
    try:
        status.update(label="Validating file...", state="running")
        file_bytes = uploaded_file.getvalue()
        validate_audio_upload(
            filename=uploaded_file.name,
            size_bytes=len(file_bytes),
            max_upload_bytes=config.max_upload_bytes,
            mime_type=uploaded_file.type,
        )

        transcription_service, summarization_service = get_services(config)

        status.update(label="Transcribing audio with Groq Whisper...", state="running")
        with temp_audio_file(file_bytes, uploaded_file.name) as audio_path:
            transcript = transcription_service.transcribe(audio_path)

        status.update(label="Analyzing transcript with Groq LLM...", state="running")
        summary = summarization_service.summarize(
            transcript, max_chars_per_chunk=config.max_chars_per_chunk
        )

        status.update(label="Done!", state="complete")
        return {"transcript": transcript, "summary": summary}

    except FileValidationError as exc:
        status.update(label="Validation failed", state="error")
        st.error(str(exc))
    except TranscriptionError as exc:
        status.update(label="Transcription failed", state="error")
        st.error(str(exc))
    except SummarizationError as exc:
        status.update(label="Summarization failed", state="error")
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - final safety net for the UI
        status.update(label="Unexpected error", state="error")
        logger.exception("Unexpected error while processing a meeting.")
        st.error("Something went wrong while processing this meeting. Please try again.")

    return None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_results(transcript: str, summary: MeetingSummary, filename: str) -> None:
    st.divider()

    with st.container(border=True):
        st.subheader("📄 Transcript")
        with st.expander("View full transcript", expanded=False):
            st.text(transcript)
        st.download_button(
            "Download transcript.txt", transcript, file_name="transcript.txt", key="dl_transcript"
        )

    with st.container(border=True):
        st.subheader("🧾 Executive Summary")
        st.write(summary.summary)

    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            st.subheader("💬 Key Discussion Points")
            if summary.discussion_points:
                for point in summary.discussion_points:
                    st.markdown(f"- {point}")
            else:
                st.caption("None recorded.")

    with col_b:
        with st.container(border=True):
            st.subheader("✅ Key Decisions")
            if summary.decisions:
                for decision in summary.decisions:
                    st.markdown(f"- {decision}")
            else:
                st.caption("None recorded.")

    with st.container(border=True):
        st.subheader("📌 Action Items")
        if summary.action_items:
            st.table(
                [
                    {
                        "Task": item.task,
                        "Owner": item.owner,
                        "Deadline": item.deadline,
                        "Priority": item.priority,
                    }
                    for item in summary.action_items
                ]
            )
        else:
            st.caption("None recorded.")

    with st.container(border=True):
        st.subheader("🔄 Unresolved / Follow-up Items")
        if summary.follow_up_items:
            for item in summary.follow_up_items:
                st.markdown(f"- {item}")
        else:
            st.caption("None recorded.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download summary.json",
            summary.to_json(),
            file_name="summary.json",
            key="dl_summary_json",
        )
    with col2:
        st.download_button(
            "Download meeting_report.md",
            summary.to_markdown(meeting_title=filename),
            file_name="meeting_report.md",
            key="dl_summary_md",
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    st.title("📝 AI Meeting Summarizer")
    st.write(
        "Upload a meeting recording to generate a transcript, summary, "
        "decisions, and action items — powered by Groq's Whisper Large V3 "
        "and a Groq-hosted LLM."
    )

    try:
        config = load_config()
    except ConfigError as exc:
        st.error(str(exc))
        return

    try:
        config.validate()
    except ConfigError as exc:
        st.error(str(exc))
        st.info("Add your API key to a `.env` file (see `.env.example`) and restart the app.")
        return

    uploaded_file = st.file_uploader(
        "Upload meeting audio",
        type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.caption(f"**{uploaded_file.name}** — {size_mb:.2f} MB")

        if size_mb > config.groq_audio_limit_mb:
            st.info(
                f"This file is larger than Groq's {config.groq_audio_limit_mb} MB "
                "per-request limit. It will automatically be split into smaller "
                "chunks with FFmpeg before transcription."
            )

        if st.button("Generate Meeting Summary", type="primary"):
            status = st.status("Starting...", expanded=True)
            result = process_meeting(uploaded_file, config, status)
            if result is not None:
                st.session_state["last_result"] = result
                st.session_state["last_filename"] = uploaded_file.name

    result = st.session_state.get("last_result")
    if result is not None:
        render_results(
            result["transcript"],
            result["summary"],
            st.session_state.get("last_filename", "Meeting Report"),
        )


if __name__ == "__main__":
    main()
