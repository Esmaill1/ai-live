# Project Context — Real-Time Voice Assistant

## What Is This?

A **real-time, conversational AI voice assistant** written in a single Python
file (`asd.py`). The assistant listens to your microphone, detects when you
speak, transcribes your words, generates an intelligent response via an LLM,
and reads it back to you using text-to-speech — all in a continuous loop.

## High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌───────────┐     ┌──────────┐
│  Microphone │────►│  Silero VAD  │────►│  Groq       │────►│  Groq     │────►│ Edge TTS │
│  (16 kHz)   │     │  (speech     │     │  Whisper    │     │  LLaMA 3  │     │ (Jenny)  │
│             │     │   detection) │     │  (STT)      │     │  (chat)   │     │          │
└─────────────┘     └──────────────┘     └─────────────┘     └───────────┘     └──────────┘
                                                                                     │
                                                                                     ▼
                                                                               ┌──────────┐
                                                                               │ Speaker  │
                                                                               │ Playback │
                                                                               └──────────┘
```

## Core Pipeline (per utterance)

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Capture** | `sounddevice.InputStream` | Captures 512-sample audio chunks at 16 kHz, mono |
| **2. VAD** | Silero VAD (PyTorch) | Each chunk is scored for speech probability (0–1). A state machine tracks speech start/end |
| **3. STT** | Groq Whisper Large v3 | When speech ends, raw audio is converted to int16 WAV bytes and sent to Groq's Whisper API |
| **4. LLM** | Groq LLaMA 3.1 8B | The transcribed text + chat history is streamed to the LLM. Tokens print to console in real time |
| **5. TTS** | Microsoft Edge TTS | The **complete** LLM response is synthesized in a single call → one continuous MP3 audio clip |
| **6. Playback** | `sounddevice.OutputStream` | MP3 is decoded to PCM via pydub and played in 1024-sample chunks |

## Key Design Decisions

### Single TTS Call (No Sentence Splitting)
The response is spoken as **one continuous audio clip** rather than split into
sentences. This avoids audible gaps/pauses between fragments. Text still
streams to the console in real-time for instant visual feedback.

### LLM Streaming in a Background Thread
The synchronous Groq streaming iterator (`for chunk in stream:`) runs in a
**background thread** via `asyncio.to_thread()`. This prevents it from
blocking the async event loop, keeping VAD and interruption handling responsive.

### Thread-Safe Interruption
`self.interrupted` is a `threading.Event`, not a plain boolean. This is
critical because interruption is **set** from the async VAD loop and **checked**
from background threads (LLM streaming, audio playback). `threading.Event` is
safe for cross-thread signaling.

### Bounded Chat History
Chat history is trimmed to the most recent 20 messages (plus the system prompt)
before each API call. This prevents exceeding the LLM context window during
long sessions.

### Task Reference Tracking
`asyncio.create_task()` results are stored in `self._tasks` with an
`add_done_callback` for cleanup. This prevents Python from silently swallowing
exceptions in fire-and-forget tasks.

## VAD State Machine

```
                 speech_prob >= 0.5
    ┌───────────────────────────────────┐
    │                                   ▼
 [IDLE]                           [TRIGGERED]
    ▲                                   │
    │       silence > 0.8s              │
    └───────────────────────────────────┘
          (process captured audio)
```

- **IDLE → TRIGGERED**: First chunk with speech probability ≥ `VAD_THRESHOLD` (0.5)
- **TRIGGERED (speech)**: Audio chunks are appended to `current_speech_buffer`
- **TRIGGERED (silence)**: Silence timer starts. Chunks still buffered (captures trailing audio)
- **TRIGGERED → IDLE**: Silence exceeds `SILENCE_DURATION` (0.8s). If total audio ≥ `MIN_SPEECH_DURATION` (0.2s), processing begins

## Interruption Flow

When the user speaks **while the AI is talking**:
1. VAD detects speech → `self.interrupted.set()`
2. `self.is_speaking = False` (immediately)
3. LLM streaming loop checks `interrupted` → stops generating
4. TTS synthesis checks `interrupted` → aborts
5. Audio playback loop checks `interrupted` → stops playing
6. The user's new speech is captured and processed normally

## File Structure

```
project/
├── asd.py                    # Main application (single-file assistant)
├── .env                      # Environment variables (GROQ_API_KEY)
├── context.md                # This file — project overview
├── developer_guide.md        # Developer guide — setup, contribute, extend
├── ai_response.mp3           # (artifact) Sample TTS output
├── my_recording.wav          # (artifact) Sample recorded audio
└── interrupt_recording.wav   # (artifact) Sample interrupted recording
```

## Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz). Standard for Whisper and Silero VAD |
| `CHANNELS` | 1 | Mono audio |
| `BLOCK_SIZE` | 512 | Samples per audio chunk sent to VAD |
| `VAD_THRESHOLD` | 0.5 | Minimum speech confidence to trigger detection |
| `SILENCE_DURATION` | 0.8 | Seconds of silence before speech is considered ended |
| `MIN_SPEECH_DURATION` | 0.2 | Minimum speech length (seconds) to process (filters noise) |
| `MAX_HISTORY` | 20 | Maximum chat messages retained (excluding system prompt) |
| `TTS_VOICE` | `en-US-JennyNeural` | Microsoft Edge TTS voice identifier |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | API key from [console.groq.com](https://console.groq.com) |

## External API Dependencies

| Service | Endpoint | Purpose | Auth |
|---------|----------|---------|------|
| **Groq** | `api.groq.com/openai/v1/audio/transcriptions` | Speech-to-text (Whisper) | API key |
| **Groq** | `api.groq.com/openai/v1/chat/completions` | LLM chat (LLaMA 3.1) | API key |
| **Edge TTS** | Microsoft Edge speech service | Text-to-speech | None (free) |

## Python Dependencies

| Package | Purpose |
|---------|---------|
| `asyncio` | Async event loop for concurrent I/O |
| `torch` | PyTorch — runs the Silero VAD model |
| `sounddevice` | Low-level audio I/O (capture + playback) |
| `numpy` | Audio data manipulation (arrays, concatenation) |
| `groq` | Official Groq Python client |
| `edge-tts` | Microsoft Edge TTS (async, free, no API key) |
| `pydub` | MP3 → PCM audio decoding |
| `scipy` | WAV file writing (for Whisper API) |
| `python-dotenv` | Load `.env` file |

> **System dependency**: `pydub` requires **ffmpeg** installed on the system for MP3 decoding.

## Threading Model

```
Main Thread (asyncio event loop)
├── process_audio_input()     — async, runs VAD state machine
├── handle_conversation()     — async task, orchestrates pipeline
├── stream_response()         — async, coordinates LLM + TTS + playback
│   ├── asyncio.to_thread(_sync_llm_stream)   — background thread for Groq streaming
│   ├── edge_tts.Communicate.stream()         — async TTS synthesis
│   └── asyncio.to_thread(play_audio_segment) — background thread for audio playback
```

## Concurrency Safety

| Shared State | Type | Access Pattern |
|-------------|------|---------------|
| `self.interrupted` | `threading.Event` | Set from async loop, checked from threads — **thread-safe** |
| `self.is_speaking` | `bool` | Set/read from async loop only — **safe** (single-threaded within loop) |
| `self.audio_queue` | `queue.Queue` | Put from sounddevice callback thread, get from async loop — **thread-safe** |
| `self.chat_history` | `list` | Modified only in async context (single-threaded) — **safe** |
| `self.running` | `bool` | Set once on shutdown — **safe** (no race) |
