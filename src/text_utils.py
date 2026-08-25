"""
Transcript cleaning and chunking utilities.

Long meetings can produce transcripts too large for a single LLM request.
Rather than truncating, we split on sentence boundaries and pack sentences
into chunks that stay under a configurable character budget.
"""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n{2,}")


def clean_transcript(raw_text: str) -> str:
    """Normalize whitespace in a raw transcript."""
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def is_effectively_empty(text: str) -> bool:
    return len(text.strip()) == 0


def needs_chunking(text: str, max_chars: int) -> bool:
    return len(text) > max_chars


def split_into_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_transcript(text: str, max_chars: int) -> list[str]:
    """
    Split `text` into chunks of at most `max_chars` characters, breaking
    only at sentence boundaries. A single sentence longer than `max_chars`
    is kept whole rather than force-split mid-word.
    """
    if not needs_chunking(text, max_chars):
        return [text] if text.strip() else []

    sentences = split_into_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current and current_len + sentence_len > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks
