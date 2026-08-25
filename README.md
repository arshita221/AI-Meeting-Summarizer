# AI Meeting Summarizer

An AI-powered meeting summarization application that converts meeting audio into a transcript and generates an action-oriented meeting report containing an executive summary, key discussion points, decisions, action items, and follow-up items.

Speech-to-text is performed using Groq's hosted **Whisper Large V3**, while summarization is performed using a Groq-hosted LLM with structured JSON output. The application is designed as a lightweight, reliable local MVP using Streamlit.

## Features

- Upload meeting recordings in MP3, WAV, M4A, MP4, WEBM, and OGG formats
- File validation for supported extensions, MIME types, file size, and empty/corrupt files
- Speech-to-text transcription using Groq **Whisper Large V3**
- Automatic FFmpeg-based chunking for audio files exceeding Groq's per-request upload limit
- Chronological reconstruction of transcripts from multiple audio chunks
- LLM-based meeting summarization using Groq
- Structured JSON output for reliable summary generation
- Executive summary generation
- Key discussion point extraction
- Key decision extraction
- Action-item extraction with:
  - Task
  - Owner
  - Deadline
  - Priority
- Follow-up and unresolved-item extraction
- Prompting designed to minimize unsupported or invented information
- Owners and deadlines default to `"Not specified"` when they are not explicitly stated
- Automatic transcript chunking for long meetings
- Safe JSON parsing and validation
- Retry handling for malformed LLM responses
- User-friendly handling of API, network, file, and processing errors
- Clean Streamlit interface
- Expandable transcript viewer
- Downloadable transcript, JSON summary, and Markdown meeting report

## Architecture

```text
Meeting Audio
      |
      v
Streamlit UI (app.py)
      |
      v
File Validation
(src/audio_utils.py)
      |
      v
Audio Chunking (if required)
        |
        |---- FFmpeg
        |
        v
Transcription Service
(src/transcription.py)
      |
      v
Groq Whisper Large V3
      |
      v
Transcript Cleaning & Chunking
(src/text_utils.py)
      |
      v
Summarization Service
(src/summarization.py)
      |
      v
Groq-hosted LLM
      |
      v
Structured JSON Response
      |
      v
Validation & Parsing
(src/schemas.py)
      |
      v
Structured Meeting Report
      |
      v
Streamlit UI
