from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import time

app = FastAPI()


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    start = time.time()

    # Step 1: just confirm we received the file (no STT yet)
    data = await audio.read()
    size_bytes = len(data)

    elapsed_ms = int((time.time() - start) * 1000)
    return JSONResponse({
        "status": "ok",
        "message": "audio received",
        "file_name": audio.filename,
        "file_size_bytes": size_bytes,
        "processing_time_ms": elapsed_ms,
        "transcript": ""  # placeholder until Step 2
    })
