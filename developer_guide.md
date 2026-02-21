# Developer Guide — AI Live Voice Assistant

> For architecture diagrams, Socket.IO protocol, and configuration reference,
> see [`context.md`](./context.md).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Web App](#running-the-web-app)
4. [Running the CLI Version](#running-the-cli-version)
5. [Project Structure Deep Dive](#project-structure-deep-dive)
6. [Server API Reference](#server-api-reference)
7. [Browser Audio Pipeline](#browser-audio-pipeline)
8. [Noise Filtering](#noise-filtering)
9. [Settings & Personalization](#settings--personalization)
10. [Interruption System](#interruption-system)
11. [How to Extend](#how-to-extend)
12. [Deployment](#deployment)
13. [Troubleshooting](#troubleshooting)
14. [Known Limitations](#known-limitations)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Uses `X \| None` union syntax |
| Groq API Key | — | Free at [console.groq.com](https://console.groq.com) |
| Modern Browser | — | Chrome, Edge, Firefox, Safari (needs `getUserMedia`) |
| ffmpeg | — | Only needed for CLI version (`asd.py`) |

---

## Installation

### 1. Create and activate a Conda environment

```bash
conda create -n groq python=3.10 -y
conda activate groq
```

### 2. Install dependencies

**For the web app** (recommended — lightweight):

```bash
pip install flask flask-socketio groq edge-tts python-dotenv
```

**For the CLI version** (heavier — requires PyTorch):

```bash
pip install torch numpy sounddevice groq edge-tts pydub scipy python-dotenv
```

### 3. Create `.env` file

```env
GROQ_API_KEY=gsk_your_key_here
```

---

## Running the Web App

```bash
conda activate groq
python server.py
```

Open **http://localhost:5000** in your browser.

### Access from other devices

```
http://<your-ip>:5000
```

> **Note**: Microphone access requires HTTPS on most browsers (except `localhost`).

---

## Running the CLI Version

```bash
conda activate groq
python asd.py
```

Requires: `torch`, `sounddevice`, `numpy`, `scipy`, `pydub`, `ffmpeg`.

---

## Project Structure Deep Dive

### `server.py` — Web App Server

```
server.py
├── Configuration
│   ├── MAX_HISTORY, TTS_VOICE, LLM_MODEL, STT_MODEL
│   ├── VOICE_LIST (25 curated Edge TTS voices)
│   ├── LANG_TO_VOICE (16 language → voice mappings)
│   └── NOISE_PHRASES (Whisper hallucination filter)
│
├── Per-session state:
│   ├── chat_histories: dict[sid → message list]
│   ├── cancel_flags: dict[sid → threading.Event]
│   ├── user_voices: dict[sid → voice name]
│   └── user_prompts: dict[sid → prompt string]
│
├── Route: GET / → render index.html
│
├── Socket.IO handlers:
│   ├── connect / disconnect — session lifecycle
│   ├── interrupt — set cancel flag
│   ├── get_voices → voices_list
│   ├── set_voice → voice_set
│   ├── set_system_prompt → prompt_set (resets history)
│   ├── restore_history → history_restored
│   └── audio_data → full pipeline:
│       ├── 1. Transcribe (Groq Whisper) + language detection
│       ├── 2. Noise filter (hallucination phrases)
│       ├── 3. LLM response (Groq GPT-OSS)
│       └── 4. TTS synthesis (Edge TTS, auto/manual voice)
│
└── Helpers:
    ├── _synthesize_tts() — async Edge TTS
    ├── _get_voice() — resolve voice (user > lang > default)
    ├── _parse_retry_after() — extract rate limit seconds
    └── _lang_code_to_name() — ISO code → display name
```

### `templates/index.html` — Web UI (Single-File SPA)

```
index.html
├── <style>  — Full CSS
│   ├── Dark theme, orb, particles, animations
│   ├── Settings panel (slide-out)
│   ├── Audio level meter
│   ├── Toast notifications
│   └── Language badge
│
├── <body>  — HTML
│   ├── Canvas (particle system)
│   ├── Mic permission banner
│   ├── Settings panel (voice, prompt, export, clear)
│   ├── Toast notification element
│   ├── Toolbar (mode toggle pill)
│   ├── Header (logo, title, v3.0)
│   ├── Language badge
│   ├── Info bar (connection, settings button)
│   ├── Central orb (icon, rings, glow, wave bars)
│   ├── Audio level meter bar
│   ├── Status label + hint
│   └── Chat panel (messages)
│
└── <script> — JavaScript
    ├── Socket.IO + event handlers
    ├── State machine (idle/listening/processing/speaking)
    ├── Interruption (triggerInterrupt)
    ├── Settings panel (open/close/save)
    ├── Mode toggle (auto-detect / push-to-talk)
    ├── Voice selector (grouped dropdown)
    ├── System prompt editor
    ├── Conversation export (.txt download)
    ├── localStorage persistence (history, voice, prompt, mode)
    ├── Rate limit countdown toast
    ├── Language detection badge
    ├── Audio capture (getUserMedia + MediaRecorder)
    ├── Browser-side VAD (RMS + consecutive frames)
    ├── Audio level meter (always-visible bar)
    ├── Keyboard shortcuts (Space, Escape)
    ├── Audio playback (HTML5 Audio from base64)
    └── Particle system (canvas animation)
```

---

## Server API Reference

### Socket.IO Events — Client → Server

| Event | Payload | Description |
|-------|---------|-------------|
| `audio_data` | `{ audio: Uint8Array }` | Send recorded audio for processing |
| `interrupt` | — | Cancel in-progress pipeline |
| `get_voices` | — | Request available TTS voices |
| `set_voice` | `{ voice: string }` | Set session TTS voice |
| `set_system_prompt` | `{ prompt: string }` | Set AI personality, resets chat |
| `restore_history` | `{ messages: [{role, content}] }` | Restore chat from localStorage |

### Socket.IO Events — Server → Client

| Event | Payload | Description |
|-------|---------|-------------|
| `state` | `{ state }` | UI state update |
| `user_message` | `{ text }` | Transcript of user speech |
| `ai_message` | `{ text }` | LLM response text |
| `audio_response` | `{ audio }` | Base64 MP3 for playback |
| `error` | `{ message }` | Error description |
| `voices_list` | `{ voices: [...] }` | TTS voice catalog |
| `voice_set` | `{ voice }` | Confirmed voice change |
| `prompt_set` | `{ prompt }` | Confirmed prompt change |
| `history_restored` | `{ count }` | Messages restored count |
| `rate_limit` | `{ message, retry_after }` | Rate limit with countdown |
| `language_detected` | `{ code, language }` | Detected spoken language |

---

## Browser Audio Pipeline

### 1. Microphone Access
```javascript
navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
})
```

### 2. Volume Analysis (Browser-Side VAD)
```javascript
analyser.getFloatTimeDomainData(data);
const rms = Math.sqrt(data.reduce((s, v) => s + v*v, 0) / data.length);
// Requires 3 consecutive frames above threshold (0.025) to start recording
```

### 3. Recording
```javascript
const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
recorder.start(100);  // 100ms chunks
// On silence (900ms) and min duration (600ms) → stop + send
```

### 4. Playback
```javascript
const audio = new Audio('data:audio/mp3;base64,' + b64);
audio.play();
```

---

## Noise Filtering

Four layers prevent background noise from triggering false responses:

| # | Layer | Where | Detail |
|---|-------|-------|--------|
| 1 | **Volume threshold** | Browser | RMS must exceed `0.025` (adjustable) |
| 2 | **Consecutive frames** | Browser | 3 consecutive above-threshold frames required |
| 3 | **Min recording duration** | Browser | Recordings under 600ms are discarded |
| 4 | **Hallucination filter** | Server | Common Whisper noise phrases are rejected |

### Whisper Hallucination Phrases (filtered server-side)

```
"thank you", "thanks", "bye", "you", "the", "okay", "hmm", "um",
"thanks for watching", "thank you for watching", "subscribe",
"like and subscribe", "music", "♪", "...", "…"
```

### Adjusting Sensitivity

In `templates/index.html`:

```javascript
const VAD_THRESHOLD = 0.035;   // Higher = less sensitive (default: 0.025)
const SILENCE_MS = 1200;       // Longer = more patience before stop (default: 900)
const MIN_RECORD_MS = 800;     // Longer = reject shorter sounds (default: 600)
const SPEECH_FRAMES_NEEDED = 5; // More frames = harder to trigger (default: 3)
```

Use the **audio level meter** (bar below the orb) to visualize your ambient noise level and calibrate the threshold.

---

## Settings & Personalization

### Voice Selector

25 curated voices across 13 languages:

| Language | Voices |
|----------|--------|
| English (US) | Jenny, Guy, Aria |
| English (UK) | Sonia, Ryan |
| English (AU) | Natasha |
| Arabic | Salma (EG), Zariyah (SA) |
| French | Denise, Henri |
| German | Katja, Conrad |
| Spanish | Elvira (ES), Dalia (MX) |
| Italian | Elsa |
| Portuguese | Francisca (BR) |
| Japanese | Nanami |
| Korean | SunHi |
| Chinese | Xiaoxiao |
| Hindi | Swara |
| Russian | Svetlana |
| Turkish | Emel |
| Dutch | Colette |
| Polish | Agnieszka |
| Swedish | Sofie |

Select via ⚙ Settings → Voice dropdown. Saved to `localStorage`.

### System Prompt

Edit the AI's personality in ⚙ Settings → System Prompt. Examples:

```
"You are a pirate assistant. Always speak like a pirate."
"You are a coding tutor. Explain concepts simply."
"أنت مساعد يتحدث العربية فقط."
```

Saving a new prompt **resets the conversation** (since the AI personality changed).

### Conversation Export

Click 📥 Export in settings to download a `.txt` file:

```
=== AI Live Conversation ===
Exported: 2/21/2026, 6:12:20 AM

[You] What's the weather like?

[AI] I'd love to help, but I don't have access to real-time weather data...
```

---

## Interruption System

| Method | Trigger | Action |
|--------|---------|--------|
| **Voice** | Speak during AI playback | VAD detects volume > threshold × 2, stops audio, records new speech |
| **Click orb** | Click during speaking/processing | Stops audio, cancels server, resets to idle |
| **Escape key** | Press Escape | Same as click orb |
| **Disconnect** | Close browser tab | Server cancels and cleans up session |

### Server-Side Cancellation Points

The `cancel_flag` (`threading.Event`) is checked at:
1. After STT transcription
2. Before LLM API call
3. After LLM response
4. Before TTS synthesis
5. During TTS — each audio chunk

---

## How to Extend

### Change the LLM Model

```python
LLM_MODEL = "llama-3.3-70b-versatile"
```

### Change the Default TTS Voice

```python
TTS_VOICE = "en-US-GuyNeural"
```

### Add More Voices to the Selector

Append to `VOICE_LIST` in `server.py`:

```python
{"name": "vi-VN-HoaiMyNeural", "locale": "Vietnamese", "gender": "Female"},
```

List all available: `edge-tts --list-voices`

### Add a Language Mapping

```python
LANG_TO_VOICE["vi"] = "vi-VN-HoaiMyNeural"
```

### Add Noise Filter Phrases

```python
NOISE_PHRASES.add("some whisper artifact")
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY server.py .
COPY templates/ templates/
COPY .env .
RUN pip install flask flask-socketio groq edge-tts python-dotenv
EXPOSE 5000
CMD ["python", "server.py"]
```

### With HTTPS (required for remote mic access)

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Quick Tunnel (ngrok)

```bash
ngrok http 5000
```

---

## Troubleshooting

### "GROQ_API_KEY is not set"
Create a `.env` file with `GROQ_API_KEY=gsk_your_key`.

### Microphone not working
- Must be on `localhost` or HTTPS
- Check browser permissions (lock icon in URL bar)
- Try Chrome (best compatibility)

### VAD too sensitive (picks up noise)
Increase `VAD_THRESHOLD` in `index.html` (see [Adjusting Sensitivity](#adjusting-sensitivity)).

### VAD not sensitive enough
Decrease `VAD_THRESHOLD` to `0.015` and `SPEECH_FRAMES_NEEDED` to `2`.

### No audio playback
- Check browser console for errors (`F12`)
- Click the orb to trigger user interaction (some browsers block autoplay)
- Check speaker volume

### Rate limit errors
Groq free tier: ~20 STT req/min, ~30 LLM req/min. The app shows a countdown toast when rate limited. Wait or upgrade plan.

### Port already in use
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <pid> /F

# Linux/Mac
lsof -i :5000 && kill <pid>
```

### Settings not saving
Clear `localStorage` in browser DevTools → Application → Local Storage → `localhost:5000`.

---

## Known Limitations

1. **No wake word** — Records all detected speech in auto-detect mode. Use push-to-talk in noisy environments.

2. **HTTPS required for remote** — Browsers need HTTPS for mic access (except `localhost`).

3. **In-memory session state** — Chat history on the server is lost on restart. Browser localStorage persists client-side.

4. **Internet required** — Groq API + Edge TTS both need internet.

5. **Echo sensitivity** — If speakers are loud and mic is nearby, AI speech can retrigger recording. Use headphones or push-to-talk mode.

6. **Single-threaded pipeline** — Each user's STT → LLM → TTS runs sequentially.

7. **API rate limits** — Groq free tier has request limits. The app handles this gracefully with retry countdowns.
