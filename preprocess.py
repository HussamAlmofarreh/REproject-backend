import io
import time
import wave
import audioop

try:
    import noisereduce as nr
    import numpy as np
except:
    nr = None
    np = None


def preprocess_audio(audio_bytes: bytes, enabled=True):
    start = time.perf_counter()

    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()

    if channels != 1:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
        channels = 1

    if rate != 16000:
        frames, _ = audioop.ratecv(
            frames,
            width,
            channels,
            rate,
            16000,
            None
        )
        rate = 16000

    if enabled and nr and np and width == 2:
        audio_np = np.frombuffer(frames, dtype=np.int16)
        reduced = nr.reduce_noise(y=audio_np, sr=rate)
        frames = reduced.astype(np.int16).tobytes()

    if enabled:
        max_amp = audioop.max(frames, width)
        if max_amp > 0:
            factor = min(30000 / max_amp, 4)
            frames = audioop.mul(frames, width, factor)

    output = io.BytesIO()

    with wave.open(output, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(frames)

    end = time.perf_counter()

    meta = {
        "preprocess_latency_ms": round((end - start) * 1000, 2),
        "preprocessing_enabled": enabled
    }

    return output.getvalue(), meta