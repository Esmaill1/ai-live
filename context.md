# Project Context — AI Live Voice Assistant

## What Is This?

A **real-time, conversational AI voice assistant** with a futuristic web UI.
The entire interaction happens in the browser — microphone capture, speech
detection, and audio playback. The server handles the AI pipeline: speech-to-text,
language model, and text-to-speech.

Two versions are included:

| File | Type | Description |
|------|------|-------------|
| `server.py` | **Web App** (primary) | Flask + Socket.IO webapp. Browser-based audio. Deployable anywhere |
| `asd.py` | CLI tool (legacy) | Uses system mic/speakers via sounddevice + Silero VAD. Local only |

## High-Level Architecture

```
┌─────────── Browser (any device) ───────────┐      ┌────────── Server (Flask) ──────────┐
│                                             │      │                                     │
│  🎤 getUserMedia → MediaRecorder            │      │  No sounddevice / No PyTorch        │
│  📊 Web Audio API AnalyserNode (VAD)        │      │  No system audio needed             │
│  🔊 HTML5 Audio (playback)                  │      │                                     │
│                                             │      │                                     │
│  audio blob ────── audio_data ─────────►    │      │  1. Groq Whisper (STT)              │
│                                             │      │  2. Groq LLaMA / GPT-OSS (LLM)     │
│  ◄──── audio_response ──── base64 MP3       │      │  3. Edge TTS (synthesis)            │
│                                             │      │                                     │
│  ◄──── state / rate_limit / lang_detected   │      │  Per-session: chat history, voice,  │
│  ──── interrupt / set_voice / set_prompt ──► │      │    system prompt, cancel flag       │
└─────────────────────────────────────────────┘      └─────────────────────────────────────┘
```

## Pipeline — What Happens Per Utterance

| Stage | Where | Component | Description |
|-------|-------|-----------|-------------|
| **1. Capture** | Browser | `getUserMedia` + `MediaRecorder` | Records audio as WebM/Opus blob |
| **2. VAD** | Browser | Web Audio API `AnalyserNode` | RMS volume with 3-frame consecutive speech detection |
| **3. Transport** | Network | Socket.IO `audio_data` event | Audio blob sent as binary via WebSocket |
| **4. STT** | Server | Groq Whisper Large v3 Turbo | Transcribes audio + detects language |
| **5. Filter** | Server | Noise phrase filter | Rejects Whisper hallucinations ("Thank you", "Bye", etc.) |
| **6. LLM** | Server | Groq GPT-OSS 120B | Generates conversational response |
| **7. TTS** | Server | Microsoft Edge TTS | Synthesizes response with auto-detected or user-selected voice |
| **8. Transport** | Network | Socket.IO `audio_response` event | Base64-encoded MP3 sent to browser |
| **9. Playback** | Browser | HTML5 `Audio` element | Plays the MP3 response |

## Key Features

### Settings Panel (⚙)
Slide-out panel with:
- **Voice selector** — 25 Edge TTS voices across 13 languages, grouped by locale
- **System prompt editor** — Customize the AI's personality without editing code
- **Conversation export** — Download chat as timestamped `.txt` file
- **Clear conversation** — Reset chat history

### Input Modes
| Mode | Activation | Behavior |
|------|-----------|----------|
| **Auto-detect** (default) | Toggle pill or settings | VAD continuously monitors volume; records when speech detected |
| **Push-to-talk** | Toggle pill or settings | Hold `Space` key or click orb to record; release to stop |

### Noise Filtering (3 layers)

| Layer | Where | What |
|-------|-------|------|
| **Volume threshold** | Browser | `VAD_THRESHOLD = 0.025` — ignores quiet ambient noise |
| **Consecutive frames** | Browser | Requires 3 above-threshold frames before recording starts |
| **Min duration** | Browser | `MIN_RECORD_MS = 600` — discards bursts under 600ms |
| **Noise phrases** | Server | Filters Whisper hallucinations: "Thank you", "Bye", "♪", "..." etc. |

### Conversation Persistence
- Chat history saved to `localStorage` on every message
- Restored to UI and server on page refresh / reconnect
- Voice preference, system prompt, and input mode also persist

### Multi-Language Auto-Detect
- Whisper detects the spoken language from audio
- TTS voice automatically switches to match (16 language mappings)
- Language badge briefly appears in the UI (e.g., "🌐 Arabic")
- User can override with manual voice selection

### Rate Limit Handling
- Server catches `groq.RateLimitError` for both STT and LLM
- Emits `rate_limit` event with countdown seconds
- Browser shows orange toast with live countdown timer

### Interruption System
Three ways to interrupt:

| Method | Action |
|--------|--------|
| **🎤 Speak** | VAD detects voice > threshold×2 → stops playback + records new speech |
| **🖱️ Click orb** | Stops playback, cancels server work, resets to idle |
| **⌨️ Escape** | Stops playback, cancels server work, resets to idle |

Server checks `cancel_flag` at 5 pipeline stages: after STT, before LLM, after LLM, before TTS, during TTS.

## Socket.IO Event Protocol

### Client → Server

| Event | Payload | Description |
|-------|---------|-------------|
| `audio_data` | `{ audio: Uint8Array }` | Recorded audio blob |
| `interrupt` | — | Cancel current AI response |
| `get_voices` | — | Request voice list |
| `set_voice` | `{ voice: string }` | Set TTS voice for session |
| `set_system_prompt` | `{ prompt: string }` | Set custom system prompt, resets history |
| `restore_history` | `{ messages: [{role, content}] }` | Restore chat from localStorage |

### Server → Client

| Event | Payload | Description |
|-------|---------|-------------|
| `state` | `{ state: string }` | `idle` / `listening` / `processing` / `speaking` |
| `user_message` | `{ text: string }` | Transcribed user speech |
| `ai_message` | `{ text: string }` | LLM response text |
| `audio_response` | `{ audio: string }` | Base64-encoded MP3 audio |
| `error` | `{ message: string }` | Error description |
| `voices_list` | `{ voices: [{name, locale, gender}] }` | Available TTS voices |
| `voice_set` | `{ voice: string }` | Confirmation of voice change |
| `prompt_set` | `{ prompt: string }` | Confirmation of prompt change |
| `history_restored` | `{ count: number }` | Number of messages restored |
| `rate_limit` | `{ message, retry_after }` | Rate limit with retry seconds |
| `language_detected` | `{ code, language }` | Detected spoken language |

## File Structure

```
project/
├── server.py                     # Web app server (Flask + Socket.IO)
├── asd.py                        # CLI version (legacy, local-only)
├── templates/
│   └── index.html                # Futuristic web UI (single-file SPA)
├── .env                          # Environment variables (GROQ_API_KEY)
├── .gitignore                    # Excludes .env, audio artifacts, __pycache__
├── context.md                    # This file — project overview
└── developer_guide.md            # Setup, API reference, extension guide
```

## Configuration Constants

### Server (`server.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_HISTORY` | 20 | Max chat messages retained per session |
| `TTS_VOICE` | `en-US-JennyNeural` | Default Edge TTS voice |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq LLM model |
| `STT_MODEL` | `whisper-large-v3-turbo` | Groq STT model |
| `VOICE_LIST` | 25 entries | Curated voices across 13 languages |
| `LANG_TO_VOICE` | 16 mappings | Language code → TTS voice |
| `NOISE_PHRASES` | ~30 entries | Whisper hallucination phrases to filter |

### Browser (`index.html`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `VAD_THRESHOLD` | 0.025 | RMS volume to detect speech |
| `SILENCE_MS` | 900 | ms of silence before recording stops |
| `MIN_RECORD_MS` | 600 | Minimum recording length to process |
| `SPEECH_FRAMES_NEEDED` | 3 | Consecutive above-threshold frames to start |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | API key from [console.groq.com](https://console.groq.com) |
| `SECRET_KEY` | No | Flask session secret (defaults to `ai-live-secret`) |

## External API Dependencies

| Service | Purpose | Auth |
|---------|---------|------|
| **Groq Whisper** | Speech-to-text + language detection | API key |
| **Groq GPT-OSS 120B** | LLM chat | API key |
| **Edge TTS** | Text-to-speech (25 voices, 13 languages) | None (free) |

## Server-Side Python Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `flask-socketio` | Real-time WebSocket communication |
| `groq` | Groq API client (STT + LLM) |
| `edge-tts` | Microsoft Edge TTS (async, free) |
| `python-dotenv` | Load `.env` file |

## UI Design

The web UI features:
- **Dark futuristic theme** with cyan/purple neon accents
- **Central glowing orb** that animates per state
- **Particle system** with connection lines
- **Glassmorphism settings panel** (slide-out from right)
- **Audio level meter** below the orb (always visible)
- **Mode toggle pill** (top-left) for auto-detect / push-to-talk
- **Language detection badge** (appears briefly for non-English)
- **Toast notifications** for settings changes and rate limits
- **Chat panel** with conversation history

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` (hold) | Record in push-to-talk mode |
| `Escape` | Interrupt AI response |

### localStorage Keys

| Key | Stores |
|-----|--------|
| `ai-live-history` | Chat messages array |
| `ai-live-voice` | Selected TTS voice name |
| `ai-live-prompt` | Custom system prompt |
| `ai-live-mode` | Input mode (`auto` / `ptt`) |
