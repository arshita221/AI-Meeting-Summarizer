# AI Meeting Summarizer (Groq Edition)

## Overview

Upload a meeting recording and get back a clean transcript plus an
action-oriented report: an executive summary, key discussion points,
decisions, and a table of action items with owner/deadline/priority —
without the LLM inventing information that wasn't actually said.

Speech-to-text runs on Groq's hosted **Whisper Large V3**, and
summarization runs on a Groq-hosted LLM using structured JSON output. The
app is a single Streamlit process — no database, no extra frameworks, no
deployment infrastructure — built for reliable local execution.

## Features

- Upload meeting audio: MP3, WAV, M4A, MP4, WEBM, OGG
- File validation: extension, MIME-type sanity check, size limit,
  empty/corrupt-file detection
- Transcription via Groq's `whisper-large-v3`
- **Automatic FFmpeg-based chunking** for audio files larger than Groq's
  per-request upload limit — chunks are transcribed independently and
  combined in the correct chronological order
- LLM summarization via a Groq chat model using JSON-mode structured
  output, safely parsed and validated (malformed responses are retried
  once, never crash the app)
- Prompting designed against hallucination: owners/deadlines default to
  "Not specified" rather than being guessed; decisions/action items are
  only recorded when the transcript actually supports them
- Automatic chunking of long **transcripts** too, with a combine pass, for
  meetings whose transcript is too large for one LLM call
- Clean, professional Streamlit UI: processing status, transcript viewer,
  summary cards, decisions, action-item table, follow-up items
- Downloads for `transcript.txt`, `summary.json`, and `meeting_report.md`
- Explicit error handling for missing/invalid API keys, unsupported files,
  oversized audio, empty transcripts, malformed LLM responses, rate limits,
  timeouts, and network failures — friendly messages, never raw stack traces
- 32 automated tests, including real FFmpeg-based audio chunking on a
  generated test tone — no Groq API key or network access required to run
  the test suite

## Architecture

```
Audio File
   |
   v
Streamlit UI (app.py)
   |
   v
File Validation (src/audio_utils.py)
   |
   v
[If file > Groq's per-request limit] --> FFmpeg segment splitting (src/audio_utils.py)
   |
   v
Transcription Service (src/transcription.py) --> Groq Whisper Large V3
   |
   v
Transcript cleaning & chunking (src/text_utils.py)
   |
   v
Summarization Service (src/summarization.py) --> Groq LLM (JSON mode)
   |
   v
Safe JSON parsing & validation (src/schemas.py)
   |
   v
Structured Meeting Report --> Streamlit UI (display + downloads)
```

## Tech Stack

| Technology | Why |
|---|---|
| **Streamlit** | Fast, single-process UI appropriate for this MVP's scope — no separate frontend/backend needed. |
| **groq (official SDK)** | Direct access to Groq's hosted Whisper Large V3 and chat models, with clear typed exceptions for error handling. |
| **FFmpeg** (system binary, via `subprocess`) | Splits oversized audio into sequential chunks with stream-copy (no re-encoding), so large recordings still work within Groq's per-request size limit. |
| **python-dotenv** | Loads `.env` locally without ever hard-coding secrets. |

No database, authentication, Docker, LangChain, or cloud deployment is
used — none of that is required for this MVP, per the project's scope.
Structured output is validated with plain Python dataclasses and manual
JSON validation (`src/schemas.py`) instead of adding a schema-validation
dependency, keeping `requirements.txt` minimal.

## Project Structure

```
meeting-summarizer-groq/
├── app.py                  # Streamlit UI + orchestration only
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── config.py            # All env-var reading & validation lives here
│   ├── audio_utils.py        # Upload validation, temp files, FFmpeg chunking
│   ├── transcription.py      # GroqTranscriptionService (Whisper Large V3)
│   ├── summarization.py      # GroqSummarizationService (JSON-mode LLM)
│   ├── prompts.py             # All prompts, kept out of service/UI code
│   ├── schemas.py             # MeetingSummary / ActionItem + safe JSON parsing
│   └── text_utils.py          # Transcript cleaning & sentence-aware chunking
└── tests/
    └── test_pipeline.py       # Full test suite, Groq calls mocked
```

## Prerequisites

- Python 3.10+
- **FFmpeg** installed and available on your system `PATH` (used for audio
  duration detection and chunking large files)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from ffmpeg.org and add it to `PATH`
- A Groq API key

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## API Key Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then open `.env` and put your key on the `GROQ_API_KEY` line:

```env
GROQ_API_KEY=your_key_here
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_LLM_MODEL=llama-3.3-70b-versatile
```

Get a key at <https://console.groq.com/keys>. **The `.env` file is
git-ignored and the key is never logged, printed, or shown in the UI.**

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key. Required. | *(none)* |
| `GROQ_WHISPER_MODEL` | Speech-to-text model. | `whisper-large-v3` |
| `GROQ_LLM_MODEL` | Chat model used for summarization. | `llama-3.3-70b-versatile` |
| `MAX_UPLOAD_MB` | Maximum accepted upload size, in MB. | `200` |
| `GROQ_AUDIO_LIMIT_MB` | Groq's per-request audio size limit — files above this are chunked automatically. | `25` |
| `MAX_CHARS_PER_CHUNK` | Character budget per summarization request before the transcript is chunked. | `12000` |

## Running the Application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Running Tests

```bash
pytest -q
```

All 32 tests run offline. Groq API calls are mocked; FFmpeg-based audio
chunking is tested against a real, locally-generated test tone (no network
access, no API key needed).

## How It Works

1. **Upload** — the user uploads a file through Streamlit; it's validated
   for extension, size, and emptiness before anything else runs.
2. **Chunking check** — if the file exceeds Groq's per-request audio size
   limit, FFmpeg splits it into sequential segments (stream-copy, no
   re-encoding) sized to stay under that limit.
3. **Transcription** — each chunk (or the whole file, if small enough) is
   sent to Groq's `whisper-large-v3`; chunk transcripts are joined back
   together in their original chronological order.
4. **Transcript processing** — whitespace is normalized and the transcript
   is checked against a configurable character budget.
5. **Summarization** — if it fits in one request, the transcript is
   summarized directly. If not, it's split at sentence boundaries, each
   chunk is summarized independently, and the partial reports are combined
   into one final report with a second LLM call.
6. **Structured output** — every LLM response is requested in JSON mode
   and parsed defensively: missing fields default to "Not specified" or
   empty lists, unusable entries are dropped, and a genuinely malformed
   response triggers one retry before a clear error is shown.
7. **Display** — the UI renders the transcript (expandable), executive
   summary, discussion points, decisions, an action-item table, and
   follow-up items, with download buttons for all three output formats.

## Prompt Design

The summarization system prompt (`src/prompts.py`) explicitly instructs
the model to:

- Use only information explicitly present in the transcript
- Never invent people, tasks, deadlines, decisions, or facts
- Return `"Not specified"` for an action item's owner/deadline when the
  transcript doesn't state one
- Distinguish discussion from decisions (only classify something as a
  decision if participants explicitly agreed) and from action items (only
  classify something as an action item if a task is actually assigned or
  committed to)
- Return a single JSON object in an exact, documented shape, with no
  extra commentary — parsed directly rather than scraped from prose

A separate, smaller prompt governs combining partial reports from chunked
transcripts, so it can be evaluated independently of the main extraction
prompt.

## Limitations

- Transcription quality depends on audio quality (background noise,
  overlapping speakers, and heavy accents can reduce accuracy).
- No speaker diarization — multiple speakers are not automatically
  separated or labeled by name.
- No real-time/streaming transcription; audio is processed after the full
  file is uploaded.
- FFmpeg-based chunking assumes roughly constant bitrate to estimate chunk
  boundaries; this is a practical approximation, not a byte-exact
  guarantee, for highly variable-bitrate files.
- Sentence-boundary transcript chunking is a regex-based heuristic, not a
  full NLP sentence splitter.
- Summarization quality is bounded by what's actually said — if a task was
  never verbally assigned, it will not appear as an action item, by
  design.
- No accuracy metrics are claimed anywhere in this project; none have been
  formally measured.
- Requires FFmpeg installed locally and a valid Groq API key with network
  access.
- No persistence — each session's results exist only until you download
  them or close the app.

## Future Improvements

- Speaker diarization
- Real-time/live meeting transcription
- Persistent storage of past meetings (a local database)
- Calendar/email integration to route action items to owners directly
- Multilingual UI and prompt localization
- A packaged FFmpeg-free fallback (pure-Python resampling) for
  environments where installing FFmpeg isn't possible

None of the above are implemented in the current codebase.
