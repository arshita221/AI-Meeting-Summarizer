"""
LLM-based summarization service using a Groq-hosted chat model.

Requests JSON output via `response_format={"type": "json_object"}` and
parses/validates the result with `src.schemas.parse_meeting_summary`,
retrying once on a malformed response before giving up cleanly.

For transcripts too large for a single request, the transcript is chunked
(see `src.text_utils`), each chunk is summarized independently, and the
partial summaries are combined into one final report with a second pass.
"""

from __future__ import annotations

import json
import logging

import groq

from src import prompts
from src.schemas import MeetingSummary, SummaryParsingError, parse_meeting_summary
from src.text_utils import chunk_transcript, clean_transcript, needs_chunking

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # one retry on a malformed/invalid response, then give up cleanly


class SummarizationError(Exception):
    """Raised when summarization fails. Message is safe to show to users."""


class GroqSummarizationService:
    """Produces a structured MeetingSummary from raw transcript text."""

    def __init__(self, api_key: str, model: str, client: groq.Groq | None = None) -> None:
        self._model = model
        self._client = client or groq.Groq(api_key=api_key)

    def summarize(self, transcript: str, max_chars_per_chunk: int) -> MeetingSummary:
        """
        Produce a MeetingSummary from `transcript`, transparently chunking
        first if it's too large for a single request.
        """
        cleaned = clean_transcript(transcript)
        if not cleaned:
            raise SummarizationError("No usable speech was detected in this recording.")

        if not needs_chunking(cleaned, max_chars_per_chunk):
            logger.info("Summarization started (single pass)")
            user_content = prompts.SINGLE_PASS_USER_TEMPLATE.format(transcript=cleaned)
            summary = self._call_llm(prompts.SYSTEM_PROMPT, user_content)
            logger.info("Summarization completed (single pass)")
            return summary

        chunks = chunk_transcript(cleaned, max_chars_per_chunk)
        logger.info("Summarization started (%d chunks)", len(chunks))

        partial_summaries: list[MeetingSummary] = []
        for i, chunk in enumerate(chunks, start=1):
            user_content = prompts.CHUNK_USER_TEMPLATE.format(
                chunk_index=i, chunk_total=len(chunks), transcript=chunk
            )
            partial_summaries.append(self._call_llm(prompts.SYSTEM_PROMPT, user_content))

        combined = self._combine_partials(partial_summaries)
        logger.info("Summarization completed (%d chunks combined)", len(chunks))
        return combined

    def _combine_partials(self, partials: list[MeetingSummary]) -> MeetingSummary:
        partial_reports_json = json.dumps([p.to_dict() for p in partials], indent=2)
        user_content = prompts.COMBINE_USER_TEMPLATE.format(
            partial_reports_json=partial_reports_json
        )
        return self._call_llm(prompts.COMBINE_SYSTEM_PROMPT, user_content)

    def _call_llm(self, system_prompt: str, user_content: str) -> MeetingSummary:
        """Call the LLM once, validating/retrying, and return a MeetingSummary."""
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
            except groq.AuthenticationError as exc:
                logger.error("Groq authentication failed during summarization.")
                raise SummarizationError(
                    "Groq authentication failed. Please check your API key."
                ) from exc
            except groq.RateLimitError as exc:
                logger.error("Groq rate limit hit during summarization.")
                raise SummarizationError(
                    "The Groq API rate limit was reached. Please try again shortly."
                ) from exc
            except groq.APITimeoutError as exc:
                logger.error("Groq request timed out during summarization.")
                raise SummarizationError(
                    "The summarization request timed out. Please try again."
                ) from exc
            except groq.APIConnectionError as exc:
                logger.error("Network error while calling the Groq chat API.")
                raise SummarizationError(
                    "Could not reach the Groq API. Please check your network connection."
                ) from exc
            except groq.BadRequestError as exc:
                logger.error("Groq rejected the summarization request as invalid.")
                raise SummarizationError(
                    "Groq rejected the summarization request. Please try again."
                ) from exc
            except groq.APIStatusError as exc:
                logger.error("Groq API returned an error status: %s", exc.status_code)
                raise SummarizationError(
                    "Summarization failed due to an API error. Please try again."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - final safety net
                logger.error("Unexpected error during summarization: %s", type(exc).__name__)
                raise SummarizationError(
                    "Summarization failed due to an unexpected error."
                ) from exc

            raw_content = self._extract_content(response)
            try:
                return parse_meeting_summary(raw_content)
            except SummaryParsingError as exc:
                last_error = exc
                logger.warning(
                    "Malformed LLM response (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    exc,
                )
                continue

        raise SummarizationError(
            "The AI response could not be parsed into a meeting report. Please try again."
        ) from last_error

    @staticmethod
    def _extract_content(response) -> str:
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""
