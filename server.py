from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from vosk import Model, KaldiRecognizer
import wave
import json
import time

app = FastAPI()

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    start = time.time()

    # Save uploaded file temporarily
    audio_bytes = await audio.read()
    temp_wav = "temp.wav"
    with open(temp_wav, "wb") as f:
        f.write(audio_bytes)

    # Open WAV file
    wf = wave.open(temp_wav, "rb")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    transcript = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript += result.get("text", "") + " "

    final_result = json.loads(rec.FinalResult())
    transcript += final_result.get("text", "")

    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "status": "ok",
        "file_name": audio.filename,
        "processing_time_ms": elapsed_ms,
        "transcript": transcript.strip()
    }
