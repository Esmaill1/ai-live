# Developer Guide — Real-Time Voice Assistant

> For a high-level overview, architecture diagrams, and configuration reference,
> see [`context.md`](./context.md).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Assistant](#running-the-assistant)
4. [Project Structure Deep Dive](#project-structure-deep-dive)
5. [Class Reference — `VoiceAssistant`](#class-reference--voiceassistant)
6. [Data Flow Walkthrough](#data-flow-walkthrough)
7. [How to Extend](#how-to-extend)
8. [Troubleshooting](#troubleshooting)
9. [Known Limitations](#known-limitations)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Uses `type | None` union syntax |
| ffmpeg | Any | Required by pydub for MP3 decoding. Must be on `PATH` |
| Microphone | — | System default input device is used |
| Speakers | — | System default output device is used |
| Groq API Key | — | Free at [console.groq.com](https://console.groq.com) |

---

## Installation

### 1. Clone / download the project

```bash
cd "c:\Users\mohamed\Desktop\Getting_Huge_Again\New folder (2)"
```

### 2. Create and activate a Conda environment (recommended)

```bash
conda create -n groq python=3.10 -y
conda activate groq
```

### 3. Install Python dependencies

```bash
pip install torch numpy sounddevice groq edge-tts pydub scipy python-dotenv
```

> **Note on PyTorch**: If you don't have a GPU or want a lighter install:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Install ffmpeg

- **Windows** (via conda): `conda install -c conda-forge ffmpeg`
- **Windows** (via choco): `choco install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 5. Create `.env` file

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Get your key from [console.groq.com/keys](https://console.groq.com/keys).

---

## Running the Assistant

```bash
conda activate groq
python asd.py
```

Expected output:

```
04:35:09 [INFO] Loading VAD model...
04:35:10 [INFO] Listening...
```

- **Speak** into your microphone — the assistant will transcribe, respond, and
  speak the answer.
- **Interrupt** by speaking while the AI is talking — it will stop and listen.
- **Exit** with `Ctrl+C`.

---

## Project Structure Deep Dive

### `asd.py` — Main Application

The entire application is in a single file, organised into clearly marked
sections within the `VoiceAssistant` class:

```
asd.py
├── Imports & constants (lines 1–45)
│   ├── Standard library: asyncio, io, logging, os, sys, threading, time, queue
│   ├── Third-party: numpy, sounddevice, torch, groq, edge_tts, pydub, scipy
│   └── Configuration constants: SAMPLE_RATE, VAD_THRESHOLD, etc.
│
├── class VoiceAssistant
│   ├── __init__()              — Setup: validate env, load VAD, init state
│   ├── _trim_history()         — Keep chat ≤ 20 messages
│   │
│   ├── # Audio Input
│   │   ├── audio_callback()        — sounddevice callback → puts chunks in queue
│   │   └── process_audio_input()   — Main VAD loop (async)
│   │
│   ├── # Conversation Pipeline
│   │   ├── handle_conversation()   — Orchestrates: transcribe → chat → speak
│   │   ├── stream_response()       — LLM streaming + TTS + playback (async)
│   │   └── _sync_llm_stream()      — Synchronous Groq streaming (runs in thread)
│   │
│   ├── # Audio Playback
│   │   └── play_audio_segment()    — Plays AudioSegment via sounddevice (in thread)
│   │
│   └── # Entry Point
│       └── run()                   — Starts the processing loop
│
└── if __name__ == "__main__"       — Instantiate and run
```

### `.env` — Environment Variables

```env
GROQ_API_KEY=gsk_...
```

Only one variable is required. The app will raise `ValueError` on startup if
it's missing.

---

## Class Reference — `VoiceAssistant`

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `running` | `bool` | `True` while the main loop should continue |
| `is_speaking` | `bool` | `True` while TTS audio is playing |
| `interrupted` | `threading.Event` | Thread-safe flag; set when user interrupts AI speech |
| `audio_queue` | `queue.Queue[np.ndarray]` | Raw audio chunks from mic callback → VAD loop |
| `groq_client` | `Groq` | Authenticated Groq API client |
| `vad_model` | `torch.nn.Module` | Silero VAD model (PyTorch) |
| `current_speech_buffer` | `list[np.ndarray]` | Accumulates audio chunks during active speech |
| `speech_start_time` | `float \| None` | Timestamp when speech started |
| `silence_start_time` | `float \| None` | Timestamp when silence started (during speech) |
| `triggered` | `bool` | VAD state: `True` = currently in speech |
| `_tasks` | `set[asyncio.Task]` | Background task references (prevents GC + exception swallowing) |
| `chat_history` | `list[dict]` | OpenAI-format chat messages |

### Methods

#### `__init__(self) -> None`
Validates `GROQ_API_KEY`, initializes the Groq client, loads the Silero VAD
model from PyTorch Hub (cached after first download), and sets up all buffers
and state.

#### `_trim_history(self) -> None`
If `chat_history` exceeds `MAX_HISTORY + 1` (system prompt + messages), trims
to keep the system prompt and the most recent `MAX_HISTORY` messages.

#### `audio_callback(self, indata, frames, time_info, status) -> None`
Called by `sounddevice` from a **separate audio thread** for each 512-sample
chunk. Copies the data and puts it in `audio_queue`. Parameter is named
`time_info` (not `time`) to avoid shadowing the `time` module.

#### `process_audio_input(self) -> None` *(async)*
The main event loop. Opens an `sd.InputStream`, then continuously:
1. Polls `audio_queue` for new audio chunks
2. Runs each chunk through Silero VAD to get speech probability
3. Manages the VAD state machine (IDLE ↔ TRIGGERED)
4. When speech ends, creates an async task to process the audio
5. Detects user interruption if `is_speaking` is True

#### `handle_conversation(self, audio_data) -> None` *(async)*
Orchestrates one full conversation turn:
1. Converts float32 audio → int16 WAV bytes
2. Sends to Groq Whisper for transcription
3. Filters out empty/noise transcriptions (< 2 chars)
4. Appends user message to chat history
5. Calls `stream_response()`

#### `stream_response(self) -> None` *(async)*
Generates and speaks the AI response:
1. Runs `_sync_llm_stream()` in a background thread → returns full response text
2. Makes a **single** Edge TTS call with the complete text
3. Collects all MP3 data from TTS
4. Decodes MP3 → PCM via pydub
5. Plays audio via `play_audio_segment()` in a background thread

Interruption is checked at every stage.

#### `_sync_llm_stream(self) -> str`
**Runs in a background thread** (not on the event loop). Calls
`groq_client.chat.completions.create()` with `stream=True`, iterates over
chunks, prints tokens to console, and accumulates the full response string.
Returns the concatenated response. Checks `interrupted` on each chunk.

#### `play_audio_segment(self, segment) -> None`
**Runs in a background thread** via `asyncio.to_thread()`. Opens an
`sd.OutputStream`, writes audio data in 1024-sample chunks, checking
`interrupted` before each write.

#### `run(self) -> None` *(async)*
Entry point. Calls `process_audio_input()` and ensures `running` is set to
`False` on exit.

---

## Data Flow Walkthrough

Here's exactly what happens when you say "Hello" to the assistant:

```
1. Microphone captures audio at 16kHz, 512 samples/chunk
   └─► audio_callback() copies to audio_queue

2. process_audio_input() polls audio_queue
   └─► torch tensor → vad_model → speech_prob = 0.87

3. speech_prob ≥ 0.5 → TRIGGERED state
   └─► chunks accumulate in current_speech_buffer

4. 0.8s of silence detected → IDLE state
   └─► full_audio = np.concatenate(buffer)  # ~1.2 seconds
   └─► asyncio.create_task(handle_conversation(full_audio))

5. handle_conversation():
   a. audio float32 → int16 → WAV bytes
   b. POST to Groq Whisper → "Hello"
   c. Append {"role": "user", "content": "Hello"} to chat_history
   d. Call stream_response()

6. stream_response():
   a. _sync_llm_stream() runs in thread:
      - POST to Groq LLaMA 3.1 (streaming)
      - Prints: "AI: Hello! How can I help you today?"
      - Returns full text

   b. edge_tts.Communicate(full_text).stream():
      - Receives MP3 data chunks
      - Concatenates into one buffer

   c. AudioSegment.from_mp3() → PCM samples

   d. play_audio_segment() runs in thread:
      - sd.OutputStream writes 1024-sample chunks
      - You hear: "Hello! How can I help you today?"
```

---

## How to Extend

### Change the LLM Model

In `_sync_llm_stream()`, replace the model string:

```python
stream = self.groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # or any Groq-supported model
    messages=self.chat_history,
    stream=True,
)
```

Available models: [Groq Models](https://console.groq.com/docs/models)

### Change the TTS Voice

Update the `TTS_VOICE` constant:

```python
TTS_VOICE = "en-US-GuyNeural"       # Male voice
TTS_VOICE = "en-GB-SoniaNeural"     # British female
TTS_VOICE = "ar-EG-SalmaNeural"     # Arabic female
```

List all voices: `edge-tts --list-voices`

### Change the System Prompt

Modify the system message in `__init__()`:

```python
self.chat_history = [
    {
        "role": "system",
        "content": "You are a pirate assistant. Always speak like a pirate.",
    }
]
```

### Add a Wake Word

In `process_audio_input()`, after transcription, check for a trigger phrase:

```python
user_text = transcription.text.strip().lower()
if not user_text.startswith("hey assistant"):
    return  # Ignore unless wake word is spoken
```

### Use a Different STT Provider

Replace the Groq Whisper call in `handle_conversation()` with any
OpenAI-compatible API:

```python
# Example: OpenAI directly
from openai import OpenAI
client = OpenAI()
transcription = client.audio.transcriptions.create(
    file=("audio.wav", byte_io.read()),
    model="whisper-1",
)
```

### Add Conversation Logging

Add a method to save conversations:

```python
import json
from datetime import datetime

def save_conversation(self):
    filename = f"conversation_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(filename, "w") as f:
        json.dump(self.chat_history, f, indent=2)
```

---

## Troubleshooting

### "GROQ_API_KEY is not set"
Create a `.env` file in the project root with `GROQ_API_KEY=gsk_your_key`.

### "No module named 'sounddevice'"
```bash
pip install sounddevice
```
On Linux, you may also need: `sudo apt install libportaudio2`

### "pydub: Couldn't find ffmpeg"
Install ffmpeg and ensure it's on your system PATH.
- Windows: `conda install -c conda-forge ffmpeg` or `choco install ffmpeg`
- macOS: `brew install ffmpeg`

### Audio input/output device errors
- Check that your mic and speakers are the **default** system device
- Try listing devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### VAD not detecting speech
- Speak louder or move closer to the mic
- Try lowering `VAD_THRESHOLD` (e.g., 0.3)
- Try increasing `SILENCE_DURATION` (e.g., 1.2) if it cuts off too early

### LLM responses are being cut off
- Increase `MAX_HISTORY` if you need more context, but be aware of token limits
- Try a model with a larger context window (e.g., `llama-3.3-70b-versatile`)

### Speech sounds robotic or low quality
- Try a different TTS voice (see "Change the TTS Voice" above)
- Edge TTS quality is generally good; if not, consider Azure Cognitive
  Services or ElevenLabs as alternatives

---

## Known Limitations

1. **No wake word** — The assistant listens and processes all detected speech.
   Implement a wake-word filter if needed (see "How to Extend").

2. **Single-user** — Only one microphone input is supported. Not designed for
   multi-user or multi-device scenarios.

3. **No persistent storage** — Chat history exists only in memory. It is lost
   when the program exits.

4. **English-optimized** — The TTS voice (`en-US-JennyNeural`) and Whisper
   model default to English. Both support multilingual input, but you'd need
   to change the TTS voice for non-English responses.

5. **Internet required** — Groq API (STT + LLM) and Edge TTS both require an
   active internet connection. There is no offline fallback.

6. **No GUI** — The application is entirely CLI-based. Audio I/O uses the
   system default devices.

7. **Response latency** — Since the full LLM response is collected before TTS
   starts, there's a brief pause between the user finishing speaking and
   hearing the response (~1–3 seconds depending on response length and
   network speed).

---

## API Rate Limits

Groq has rate limits on their free tier:

| Endpoint | Limit |
|----------|-------|
| Whisper (STT) | 20 requests/min |
| Chat Completions (LLM) | 30 requests/min |

If you hit rate limits, add a small delay between conversation turns or
upgrade to a paid Groq plan.
