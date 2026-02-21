"""
server.py — Fully browser-based Voice Assistant web app.

Audio capture and playback happen entirely in the browser.
The server handles: STT (Groq Whisper) → LLM (Groq LLaMA) → TTS (Edge TTS).

Features: per-session voices, custom system prompts, multi-language auto-detect,
rate limit handling, conversation restore.
"""

import asyncio
import base64
import json
import logging
import os
import threading
import time

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from groq import Groq, RateLimitError
import edge_tts
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
MAX_HISTORY = 20
TTS_VOICE = "en-US-JennyNeural"
LLM_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"

# Curated list of TTS voices (subset of Edge TTS)
VOICE_LIST = [
    {"name": "en-US-JennyNeural", "locale": "English (US)", "gender": "Female"},
    {"name": "en-US-GuyNeural", "locale": "English (US)", "gender": "Male"},
    {"name": "en-US-AriaNeural", "locale": "English (US)", "gender": "Female"},
    {"name": "en-GB-SoniaNeural", "locale": "English (UK)", "gender": "Female"},
    {"name": "en-GB-RyanNeural", "locale": "English (UK)", "gender": "Male"},
    {"name": "en-AU-NatashaNeural", "locale": "English (AU)", "gender": "Female"},
    {"name": "ar-EG-SalmaNeural", "locale": "Arabic (Egypt)", "gender": "Female"},
    {"name": "ar-SA-ZariyahNeural", "locale": "Arabic (Saudi)", "gender": "Female"},
    {"name": "fr-FR-DeniseNeural", "locale": "French", "gender": "Female"},
    {"name": "fr-FR-HenriNeural", "locale": "French", "gender": "Male"},
    {"name": "de-DE-KatjaNeural", "locale": "German", "gender": "Female"},
    {"name": "de-DE-ConradNeural", "locale": "German", "gender": "Male"},
    {"name": "es-ES-ElviraNeural", "locale": "Spanish (Spain)", "gender": "Female"},
    {"name": "es-MX-DaliaNeural", "locale": "Spanish (Mexico)", "gender": "Female"},
    {"name": "it-IT-ElsaNeural", "locale": "Italian", "gender": "Female"},
    {"name": "pt-BR-FranciscaNeural", "locale": "Portuguese (BR)", "gender": "Female"},
    {"name": "ja-JP-NanamiNeural", "locale": "Japanese", "gender": "Female"},
    {"name": "ko-KR-SunHiNeural", "locale": "Korean", "gender": "Female"},
    {"name": "zh-CN-XiaoxiaoNeural", "locale": "Chinese (Mandarin)", "gender": "Female"},
    {"name": "hi-IN-SwaraNeural", "locale": "Hindi", "gender": "Female"},
    {"name": "ru-RU-SvetlanaNeural", "locale": "Russian", "gender": "Female"},
    {"name": "tr-TR-EmelNeural", "locale": "Turkish", "gender": "Female"},
    {"name": "nl-NL-ColetteNeural", "locale": "Dutch", "gender": "Female"},
    {"name": "pl-PL-AgnieszkaNeural", "locale": "Polish", "gender": "Female"},
    {"name": "sv-SE-SofieNeural", "locale": "Swedish", "gender": "Female"},
]

# Language code → default TTS voice (for auto-detect)
LANG_TO_VOICE = {
    "en": "en-US-JennyNeural",
    "ar": "ar-EG-SalmaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "hi": "hi-IN-SwaraNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-AgnieszkaNeural",
    "sv": "sv-SE-SofieNeural",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, conversational AI voice assistant. "
    "Keep responses concise and natural for speech. "
    "Respond in 2-3 sentences when possible."
)

# ── Validate ──────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Create a .env file.")

groq_client = Groq(api_key=api_key)

# ── Flask + Socket.IO ─────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "ai-live-secret")
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=16 * 1024 * 1024,  # 16 MB for audio blobs
)

# Per-session state
chat_histories: dict[str, list[dict[str, str]]] = {}
cancel_flags: dict[str, threading.Event] = {}
user_voices: dict[str, str] = {}       # per-session TTS voice override
user_prompts: dict[str, str] = {}      # per-session system prompt override


def _get_system_prompt(sid: str) -> dict:
    """Get the system prompt for a session."""
    content = user_prompts.get(sid, DEFAULT_SYSTEM_PROMPT)
    return {"role": "system", "content": content}


def _get_voice(sid: str, detected_lang: str | None = None) -> str:
    """Get TTS voice: user override > auto-detected language > default."""
    if sid in user_voices:
        return user_voices[sid]
    if detected_lang and detected_lang in LANG_TO_VOICE:
        return LANG_TO_VOICE[detected_lang]
    return TTS_VOICE


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Socket.IO Events ─────────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    sid = request.sid
    sys_prompt = _get_system_prompt(sid)
    chat_histories[sid] = [sys_prompt]
    cancel_flags[sid] = threading.Event()
    logger.info("Client connected: %s", sid)
    emit("state", {"state": "idle"})


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    chat_histories.pop(sid, None)
    user_voices.pop(sid, None)
    user_prompts.pop(sid, None)
    flag = cancel_flags.pop(sid, None)
    if flag:
        flag.set()
    logger.info("Client disconnected: %s", sid)


@socketio.on("interrupt")
def handle_interrupt():
    """Client detected user speech during AI response — cancel everything."""
    sid = request.sid
    flag = cancel_flags.get(sid)
    if flag:
        flag.set()
    logger.info("[%s] Interrupted by user", sid)


# ── Voice & Settings Events ──────────────────────────────────────────
@socketio.on("get_voices")
def handle_get_voices():
    """Send the curated voice list to the client."""
    emit("voices_list", {"voices": VOICE_LIST})


@socketio.on("set_voice")
def handle_set_voice(data):
    """Set the TTS voice for this session."""
    sid = request.sid
    voice = data.get("voice", TTS_VOICE)
    user_voices[sid] = voice
    logger.info("[%s] Voice set to: %s", sid, voice)
    emit("voice_set", {"voice": voice})


@socketio.on("set_system_prompt")
def handle_set_system_prompt(data):
    """Set a custom system prompt and reset chat history."""
    sid = request.sid
    prompt = data.get("prompt", "").strip()
    if not prompt:
        prompt = DEFAULT_SYSTEM_PROMPT

    user_prompts[sid] = prompt
    sys_prompt = _get_system_prompt(sid)
    chat_histories[sid] = [sys_prompt]
    logger.info("[%s] System prompt updated, history reset", sid)
    emit("prompt_set", {"prompt": prompt})


@socketio.on("restore_history")
def handle_restore_history(data):
    """Restore chat history from client's localStorage."""
    sid = request.sid
    messages = data.get("messages", [])
    if not messages:
        return

    sys_prompt = _get_system_prompt(sid)
    history = [sys_prompt]

    for msg in messages[-MAX_HISTORY:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})

    chat_histories[sid] = history
    logger.info("[%s] Restored %d messages from client", sid, len(history) - 1)
    emit("history_restored", {"count": len(history) - 1})


# ── Main Audio Pipeline ──────────────────────────────────────────────
@socketio.on("audio_data")
def handle_audio(data):
    """Receive audio from browser → STT → LLM → TTS → send audio back."""
    sid = request.sid
    audio_bytes = data.get("audio")

    if not audio_bytes:
        emit("state", {"state": "idle"})
        return

    # Reset cancellation flag for this new request
    flag = cancel_flags.get(sid)
    if flag:
        flag.clear()
    else:
        flag = threading.Event()
        cancel_flags[sid] = flag

    emit("state", {"state": "processing"})

    # ── 1. Transcribe (Groq Whisper) ──────────────────────────────────
    detected_lang = None
    try:
        logger.info("[%s] Transcribing audio (%d bytes)...", sid, len(audio_bytes))
        transcription = groq_client.audio.transcriptions.create(
            file=("recording.webm", audio_bytes),
            model=STT_MODEL,
            response_format="verbose_json",
        )
        user_text = transcription.text.strip()
        # Extract detected language
        detected_lang = getattr(transcription, "language", None)
    except RateLimitError as e:
        retry_after = _parse_retry_after(e)
        logger.warning("[%s] Rate limited on STT, retry after %ds", sid, retry_after)
        emit("rate_limit", {"message": "STT rate limited", "retry_after": retry_after})
        emit("state", {"state": "idle"})
        return
    except Exception as e:
        logger.error("Transcription error: %s", e)
        emit("error", {"message": f"Transcription failed: {e}"})
        emit("state", {"state": "idle"})
        return

    if flag.is_set():
        logger.info("[%s] Cancelled after transcription", sid)
        emit("state", {"state": "idle"})
        return

    # Filter empty / noise — Whisper often hallucinates short phrases from noise
    NOISE_PHRASES = {
        "thank you", "thanks", "bye", "you", "the", "a", "i", "it",
        "okay", "ok", "hmm", "hm", "um", "uh", "ah", "oh",
        "yeah", "yes", "no", "so", "and", "but", "or", "is", "was",
        "thank you.", "thanks.", "bye.", "the end", "the end.",
        "thanks for watching", "thanks for watching.",
        "thank you for watching", "thank you for watching.",
        "subscribe", "like and subscribe",
        "music", "♪", "...", "…",
    }
    if not user_text or len(user_text) < 3 or user_text.lower().strip('.!?, ') in NOISE_PHRASES:
        logger.info("[%s] Filtered out noise: '%s'", sid, user_text)
        emit("state", {"state": "idle"})
        return

    logger.info("[%s] User: %s", sid, user_text)
    emit("user_message", {"text": user_text})

    # Emit detected language
    if detected_lang:
        lang_name = _lang_code_to_name(detected_lang)
        emit("language_detected", {"code": detected_lang, "language": lang_name})

    # Update history
    sys_prompt = _get_system_prompt(sid)
    history = chat_histories.get(sid, [sys_prompt])
    history.append({"role": "user", "content": user_text})

    # ── 2. LLM Response ──────────────────────────────────────────────
    if flag.is_set():
        logger.info("[%s] Cancelled before LLM", sid)
        emit("state", {"state": "idle"})
        return

    try:
        logger.info("[%s] Generating LLM response...", sid)
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=history,
        )
        ai_text = response.choices[0].message.content.strip()
    except RateLimitError as e:
        retry_after = _parse_retry_after(e)
        logger.warning("[%s] Rate limited on LLM, retry after %ds", sid, retry_after)
        emit("rate_limit", {"message": "LLM rate limited", "retry_after": retry_after})
        emit("state", {"state": "idle"})
        return
    except Exception as e:
        logger.error("LLM error: %s", e)
        emit("error", {"message": f"LLM failed: {e}"})
        emit("state", {"state": "idle"})
        return

    if flag.is_set():
        logger.info("[%s] Cancelled after LLM", sid)
        emit("state", {"state": "idle"})
        return

    logger.info("[%s] AI: %s", sid, ai_text[:80])
    history.append({"role": "assistant", "content": ai_text})

    # Trim history
    if len(history) > MAX_HISTORY + 1:
        history = [history[0]] + history[-(MAX_HISTORY):]
    chat_histories[sid] = history

    emit("ai_message", {"text": ai_text})

    # ── 3. TTS Synthesis (Edge TTS) ───────────────────────────────────
    if flag.is_set():
        logger.info("[%s] Cancelled before TTS", sid)
        emit("state", {"state": "idle"})
        return

    voice = _get_voice(sid, detected_lang)
    emit("state", {"state": "speaking"})
    try:
        logger.info("[%s] Synthesizing TTS with voice=%s...", sid, voice)
        loop = asyncio.new_event_loop()
        mp3_data = loop.run_until_complete(_synthesize_tts(ai_text, voice, flag))
        loop.close()

        if flag.is_set():
            logger.info("[%s] Cancelled during TTS", sid)
            emit("state", {"state": "idle"})
            return

        audio_b64 = base64.b64encode(mp3_data).decode("utf-8")
        emit("audio_response", {"audio": audio_b64})
        logger.info("[%s] Sent %d bytes of audio", sid, len(mp3_data))

    except Exception as e:
        logger.error("TTS error: %s", e)
        emit("error", {"message": f"TTS failed: {e}"})
        emit("state", {"state": "idle"})


# ── Helpers ───────────────────────────────────────────────────────────
async def _synthesize_tts(text: str, voice: str, cancel: threading.Event) -> bytes:
    """Synthesize text to MP3 bytes using Edge TTS, with cancellation."""
    communicate = edge_tts.Communicate(text, voice)
    mp3_data = b""
    async for chunk in communicate.stream():
        if cancel.is_set():
            break
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]
    return mp3_data


def _parse_retry_after(error: RateLimitError) -> int:
    """Extract retry-after seconds from a rate limit error."""
    try:
        # Try to get from response headers
        if hasattr(error, "response") and error.response:
            val = error.response.headers.get("retry-after", "30")
            return int(float(val))
    except (ValueError, AttributeError):
        pass
    return 30  # default 30 seconds


LANG_NAMES = {
    "en": "English", "ar": "Arabic", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "hi": "Hindi", "ru": "Russian",
    "tr": "Turkish", "nl": "Dutch", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "no": "Norwegian", "el": "Greek",
    "he": "Hebrew", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
    "ms": "Malay", "uk": "Ukrainian", "cs": "Czech", "ro": "Romanian",
}


def _lang_code_to_name(code: str) -> str:
    """Convert ISO 639-1 code to human-readable language name."""
    return LANG_NAMES.get(code, code.upper())


# ── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting AI Live at http://localhost:5000")
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
