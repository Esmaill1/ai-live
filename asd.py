import asyncio
import io
import logging
import os
import sys
import threading
import time
import queue

import numpy as np
import sounddevice as sd
import torch
from groq import Groq
import edge_tts
from pydub import AudioSegment
from scipy.io.wavfile import write as wav_write
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Audio
SAMPLE_RATE = 16000  # Standard for VAD and Whisper
CHANNELS = 1
BLOCK_SIZE = 512  # Audio chunk size for VAD processing

# VAD Parameters
VAD_THRESHOLD = 0.5          # Confidence threshold for speech
SILENCE_DURATION = 0.8       # Seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 0.2    # Min seconds to consider it valid speech

# Chat
MAX_HISTORY = 20  # Keep last N messages (+ system prompt)

# TTS
TTS_VOICE = "en-US-JennyNeural"


class VoiceAssistant:
    def __init__(self) -> None:
        self.running = True
        self.is_speaking = False
        # Thread-safe interruption flag
        self.interrupted = threading.Event()
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        # --- Validate env vars ---
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Create a .env file with GROQ_API_KEY=your_key"
            )

        # API Clients
        self.groq_client = Groq(api_key=api_key)

        # Load Silero VAD
        logger.info("Loading VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

        # Audio Buffers
        self.current_speech_buffer: list[np.ndarray] = []
        self.speech_start_time: float | None = None
        self.silence_start_time: float | None = None
        self.triggered = False

        # Active background tasks (prevent fire-and-forget warnings)
        self._tasks: set[asyncio.Task] = set()

        # Chat History
        self.chat_history: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, conversational AI voice assistant. "
                    "Keep responses concise and natural for speech."
                ),
            }
        ]

    # ------------------------------------------------------------------
    # Chat history management
    # ------------------------------------------------------------------
    def _trim_history(self) -> None:
        """Keep chat history bounded to avoid exceeding the LLM context window."""
        # +1 accounts for the system prompt at index 0
        if len(self.chat_history) > MAX_HISTORY + 1:
            self.chat_history = (
                [self.chat_history[0]] + self.chat_history[-(MAX_HISTORY):]
            )

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------
    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Input callback for sounddevice.

        Note: the parameter is called *time_info* (not *time*) to avoid
        shadowing the built-in ``time`` module.
        """
        if status:
            logger.warning("Audio input status: %s", status)
        self.audio_queue.put(indata.copy())

    async def process_audio_input(self) -> None:
        """Continuously process audio for VAD."""
        logger.info("Listening...")

        with sd.InputStream(
            callback=self.audio_callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
        ):
            while self.running:
                try:
                    chunk = self.audio_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                # Silero expects float32 tensor
                audio_tensor = torch.from_numpy(chunk.flatten()).float()

                # Get speech confidence
                speech_prob = self.vad_model(audio_tensor, SAMPLE_RATE).item()

                # --- VAD State Machine ---
                if speech_prob >= VAD_THRESHOLD:
                    # Speech detected
                    if not self.triggered:
                        self.triggered = True
                        self.current_speech_buffer = []

                    self.current_speech_buffer.append(chunk)
                    self.silence_start_time = None

                    # Handle interruption
                    if self.is_speaking:
                        logger.info("Interruption detected!")
                        self.interrupted.set()
                        self.is_speaking = False

                elif self.triggered:
                    # Speech ongoing but currently silent (pause)
                    self.current_speech_buffer.append(chunk)

                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()

                    # Check if silence exceeded threshold
                    if time.time() - self.silence_start_time > SILENCE_DURATION:
                        self.triggered = False

                        # Process the captured audio
                        full_audio = np.concatenate(self.current_speech_buffer)
                        duration = len(full_audio) / SAMPLE_RATE

                        if duration >= MIN_SPEECH_DURATION:
                            # Save task reference so exceptions are not silently lost
                            task = asyncio.create_task(
                                self.handle_conversation(full_audio)
                            )
                            self._tasks.add(task)
                            task.add_done_callback(self._tasks.discard)

                        self.current_speech_buffer = []
                        self.silence_start_time = None

    # ------------------------------------------------------------------
    # Conversation pipeline
    # ------------------------------------------------------------------
    async def handle_conversation(self, audio_data: np.ndarray) -> None:
        """Transcribe -> Chat -> Speak"""
        logger.info("Processing audio...")

        # 1. Transcribe -----------------------------------------------
        # Convert numpy float32 -> int16 wav bytes for Whisper
        audio_int16 = (audio_data * 32767).astype(np.int16)
        byte_io = io.BytesIO()
        wav_write(byte_io, SAMPLE_RATE, audio_int16)
        byte_io.seek(0)

        try:
            transcription = self.groq_client.audio.transcriptions.create(
                file=("audio.wav", byte_io.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
            user_text = transcription.text.strip()
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return

        # Filter empty / noise
        if not user_text or len(user_text) < 2:
            return

        logger.info("User: %s", user_text)
        self.chat_history.append({"role": "user", "content": user_text})
        self._trim_history()

        # 2. LLM Stream & TTS -----------------------------------------
        await self.stream_response()

    async def stream_response(self) -> None:
        """Stream LLM text to console, then speak the full response at once.

        Text streams to the terminal in real-time for instant visual
        feedback.  Once the full response is collected, a single TTS call
        synthesises the entire text — producing one continuous audio clip
        with zero inter-sentence pauses.
        """
        self.is_speaking = True
        self.interrupted.clear()

        try:
            # --- 1. Stream LLM tokens (runs in thread, won't block loop) ---
            full_response = await asyncio.to_thread(self._sync_llm_stream)

            if not full_response or self.interrupted.is_set():
                return

            self.chat_history.append(
                {"role": "assistant", "content": full_response}
            )
            self._trim_history()

            # --- 2. One single TTS call for the whole response ---
            if self.interrupted.is_set():
                return

            communicate = edge_tts.Communicate(full_response, TTS_VOICE)
            mp3_data = b""
            async for tts_chunk in communicate.stream():
                if self.interrupted.is_set():
                    return
                if tts_chunk["type"] == "audio":
                    mp3_data += tts_chunk["data"]

            if self.interrupted.is_set():
                return

            seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            await asyncio.to_thread(self.play_audio_segment, seg)

        except Exception as e:
            logger.error("Error in stream: %s", e)
        finally:
            self.is_speaking = False
            print()  # Newline after AI response

    def _sync_llm_stream(self) -> str:
        """Stream LLM tokens to console (synchronous, runs in a thread)."""
        stream = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=self.chat_history,
            stream=True,
        )

        full_response = ""
        print("AI: ", end="", flush=True)

        for chunk in stream:
            if self.interrupted.is_set():
                print("\n[Stopped]")
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            print(delta, end="", flush=True)
            full_response += delta

        return full_response

    # ------------------------------------------------------------------
    # Audio playback
    # ------------------------------------------------------------------
    def play_audio_segment(self, segment: AudioSegment) -> None:
        """Plays an AudioSegment using sounddevice (runs in a thread)."""
        if self.interrupted.is_set():
            return

        try:
            samples = np.array(segment.get_array_of_samples())

            if segment.channels == 2:
                samples = samples.reshape((-1, 2))

            with sd.OutputStream(
                samplerate=segment.frame_rate,
                channels=segment.channels,
                dtype='int16',
            ) as out_stream:
                chunk_size = 1024
                for i in range(0, len(samples), chunk_size):
                    if self.interrupted.is_set():
                        break
                    out_stream.write(samples[i:i + chunk_size])
        except Exception as e:
            logger.error("Playback error: %s", e)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Start the assistant processing loop."""
        try:
            await self.process_audio_input()
        finally:
            self.running = False
            logger.info("Assistant stopped.")


if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        assistant.running = False
        logger.info("Exiting...")
