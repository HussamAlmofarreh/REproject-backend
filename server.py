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
from response_generator import process_nlu_result

# Temporary, until we connect to cloud API
CLOUD_AVAILABLE = False

app = FastAPI()

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

# ======== Settings ========
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_DURATION_MS = 10_000  # 10 seconds
REQUIRED_SR = 16_000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2  # 16-bit PCM
LOG_PATH = "logs.jsonl"
LOG_TRANSCRIPT = False  # privacy: off by default

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
    t_pre_end = t0
    t_stt_end = t0

    intent = "UNKNOWN"
    slots = {}
    nlu_confidence = 0.0
    cloud_fallback = False
    mode = "local"

    policy_action = ""
    response_text = ""

    try:
        audio_bytes = await audio.read()

        # ---- size limit
        if len(audio_bytes) > MAX_BYTES:
            raise ValueError("too_large")

        # ---- open WAV from memory for validation
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

        # ---- preprocessing
        audio_bytes, sr = preprocess_audio(audio_bytes)
        t_pre_end = time.perf_counter()

        # ---- reopen processed audio for STT
        try:
            wf = wave.open(io.BytesIO(audio_bytes), "rb")
        except wave.Error:
            raise ValueError("preprocess_invalid_wav")
        sr = wf.getframerate()

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

                if "result" in result:
                    word_confs.extend([w.get("conf", 0.0) for w in result["result"]])

        final_result = json.loads(rec.FinalResult())
        transcript += final_result.get("text", "")

        if "result" in final_result:
            word_confs.extend([w.get("conf", 0.0) for w in final_result["result"]])

        transcript = transcript.strip()
        stt_confidence = sum(word_confs) / len(word_confs) if word_confs else 0.0
        t_stt_end = time.perf_counter()

        # ---- NLU
        nlu_result = recognize_intent(transcript)
        intent = nlu_result["intent"]
        slots = nlu_result["slots"]
        nlu_confidence = nlu_result["confidence"]

        # ---- policy gate + response generation
        processed_nlu = process_nlu_result(nlu_result)
        policy_action = processed_nlu["policy_action"]
        response_text = processed_nlu["response_text"]

        # ---- fallback decision logic
        FALLBACK_STT_THRESHOLD = 0.6
        FALLBACK_NLU_THRESHOLD = 0.5

        if (
            intent == "UNKNOWN"
            or stt_confidence < FALLBACK_STT_THRESHOLD
            or nlu_confidence < FALLBACK_NLU_THRESHOLD
        ):
            cloud_fallback = True

            if CLOUD_AVAILABLE:
                mode = "cloud"
            else:
                mode = "degraded"

        # ---- degraded mode overrides normal response
        if mode == "degraded":
            response_text = "I didn't catch that clearly. Could you repeat?"
        elif intent == "UNKNOWN" and not response_text:
            response_text = "Sorry, I didn't understand that."

    except ValueError as e:
        status = "error"
        errors.append(str(e))
        t_validate_end = time.perf_counter()
        t_pre_end = t_validate_end
        t_stt_end = t_validate_end

    except Exception as e:
        status = "error"
        errors.append("internal_error")
        t_stt_end = time.perf_counter()
        print("UNEXPECTED ERROR:", repr(e))

    t_end = time.perf_counter()
    latency_ms = {
        "validate": round((t_validate_end - t0) * 1000, 2),
        "preprocess": round((t_pre_end - t_validate_end) * 1000, 2),
        "stt": round((t_stt_end - t_pre_end) * 1000, 2),
        "total": round((t_end - t0) * 1000, 2),
    }

    response = {
        "request": {
            "id": request_id,
            "status": status,
            "filename": audio.filename,
        },
        "audio": {
            "duration_ms": duration_ms,
        },
        "stt": {
            "transcript": transcript if status == "ok" else "",
            "confidence": round(stt_confidence, 4) if status == "ok" else 0.0,
        },
        "nlu": {
            "intent": intent if status == "ok" else "UNKNOWN",
            "confidence": nlu_confidence if status == "ok" else 0.0,
            "slots": slots if status == "ok" else {},
        },
        "system": {
            "mode": mode if status == "ok" else "error",
            "cloud_fallback": cloud_fallback if status == "ok" else False,
        },
        "response_generator": {
            "policy_action": policy_action if status == "ok" else "",
            "response_text": response_text if status == "ok" else "",
        },
        "latency_ms": latency_ms,
        "errors": errors,
    }

    # ---- metadata logging (privacy-aware)
    log_record = json.loads(json.dumps(response))
    if not LOG_TRANSCRIPT and "stt" in log_record:
        log_record["stt"].pop("transcript", None)
    log_event(log_record)

    return response
