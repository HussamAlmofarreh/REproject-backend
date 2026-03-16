from fastapi import FastAPI, UploadFile, File
from vosk import Model, KaldiRecognizer
import wave
import json
import time
import io
from uuid import uuid4
from datetime import datetime, timezone
from threading import Lock
from nlu.intent_recognizer import recognize_intent
from preprocess import preprocess_audio


app = FastAPI()

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

# ======== Settings ========
MAX_BYTES = 5 * 1024 * 1024       # 5 MB
MAX_DURATION_MS = 10_000          # 10 seconds
REQUIRED_SR = 16_000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2            # 16-bit PCM
LOG_PATH = "logs.jsonl"
LOG_TRANSCRIPT = False            # privacy: off by default

_lock = Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: dict) -> None:
    event = dict(event)
    event.setdefault("ts", utc_now_iso())
    line = json.dumps(event, ensure_ascii=False)
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def extract_confidence(vosk_result: dict) -> float:
    words = vosk_result.get("result", [])
    if not words:
        return 0.0
    return sum(w.get("conf", 0.0) for w in words) / max(len(words), 1)


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    request_id = str(uuid4())
    t0 = time.perf_counter()

    transcript = ""
    stt_confidence = 0.0
    duration_ms = 0
    errors = []
    status = "ok"

    t_validate_end = t0
    t_pre_end = t0  # preprocessing not implemented yet
    t_stt_end = t0

    intent = "UNKNOWN"
    slots = {}
    nlu_confidence = 0.0

    try:
        audio_bytes = await audio.read()

        # ---- size limit
        if len(audio_bytes) > MAX_BYTES:
            raise ValueError("too_large")

        # ---- open wav from memory (no temp file)
        try:
            wf = wave.open(io.BytesIO(audio_bytes), "rb")
        except wave.Error:
            raise ValueError("invalid_wav")

        # ---- validate format
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        duration_ms = int((nframes / float(sr)) * 1000) if sr else 0

        if sr != REQUIRED_SR:
            raise ValueError("bad_sample_rate")
        if ch != REQUIRED_CHANNELS:
            raise ValueError("bad_channels")
        if sw != REQUIRED_SAMPWIDTH:
            raise ValueError("bad_bit_depth")
        if duration_ms > MAX_DURATION_MS:
            raise ValueError("too_long")

        t_validate_end = time.perf_counter()
        audio_bytes = preprocess_audio(audio_bytes)
        t_pre_end = time.perf_counter()

        # ---- STT
        rec = KaldiRecognizer(model, sr)
        rec.SetWords(True)

        word_confs = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcript += result.get("text", "") + " "
                # collect confidence values if present
                if "result" in result:
                    word_confs.extend([w.get("conf", 0.0)
                                      for w in result["result"]])

        final_result = json.loads(rec.FinalResult())
        transcript += final_result.get("text", "")

        if "result" in final_result:
            word_confs.extend([w.get("conf", 0.0)
                              for w in final_result["result"]])

        transcript = transcript.strip()
        stt_confidence = (sum(word_confs) / len(word_confs)
                          ) if word_confs else 0.0

        nlu_result = recognize_intent(transcript)
        intent = nlu_result["intent"]
        slots = nlu_result["slots"]
        nlu_confidence = nlu_result["confidence"]

        t_stt_end = time.perf_counter()

    except ValueError as e:
        status = "error"
        errors.append(str(e))
        t_validate_end = time.perf_counter()
        t_pre_end = t_validate_end
        t_stt_end = t_validate_end

    except Exception:
        status = "error"
        errors.append("internal_error")
        t_stt_end = time.perf_counter()

    t_end = time.perf_counter()

    latency_ms = {
        "validate": round((t_validate_end - t0) * 1000, 2),
        "preprocess": round((t_pre_end - t_validate_end) * 1000, 2),
        "stt": round((t_stt_end - t_pre_end) * 1000, 2),
        "total": round((t_end - t0) * 1000, 2),
    }

    response = {
        "request_id": request_id,
        "status": status,
        "filename": audio.filename,
        "duration_ms": duration_ms,
        "transcript": transcript if status == "ok" else "",
        "stt_confidence": round(stt_confidence, 4) if status == "ok" else 0.0,
        "intent": intent if status == "ok" else "UNKNOWN",
        "slots": slots if status == "ok" else {},
        "nlu_confidence": nlu_confidence if status == "ok" else 0.0,
        "latency_ms": latency_ms,
        "errors": errors,
    }

    # ---- metadata logging (privacy-aware)
    log_record = dict(response)
    if not LOG_TRANSCRIPT:
        log_record.pop("transcript", None)
    log_event(log_record)

    return response
