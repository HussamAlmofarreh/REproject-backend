# reproject-backend

## Speech-to-Text (FastAPI + Vosk)

This project includes a simple Speech-to-Text (STT) backend using **FastAPI** and **Vosk**.
It exposes an HTTP endpoint that accepts a WAV file and returns a transcript.

### Requirements

- Python 3.9+
- Vosk model (download separately)
- FFmpeg (recommended for converting audio into the correct WAV format)

### Setup

1. **Create and activate a Python environment** (recommended)

```bash
conda create -n stt python=3.9 -y
conda activate stt
```
2. Install Python dependencies
```
pip install fastapi uvicorn vosk python-multipart requests
```
4.	Download a Vosk model
Download an English model from Vosk and place it inside the models/ folder.
Note: Models are large and are not committed to the repo.

5.	Run the server
uvicorn server:app --reload


Testing with the provided client
Run: python client.py path/to/audio.wav --pretty



WAV format requirements (important)

Vosk works best with:
	•	Mono (1 channel)
	•	16kHz sample rate
	•	PCM WAV (not AAC/ALAC inside a .wav container, and not WAV_EXTENSIBLE)

If your file fails with errors like file does not start with RIFF id or unknown format: 65534,
convert it with FFmpeg:  ffmpeg -i input.wav -ac 1 -ar 16000 -c:a pcm_s16le output.wav

Notes
	•	models/ (Vosk model) and samples/ (audio test files) are excluded from Git via .gitignore.
	•	If you want to test audio files, create a local samples/ folder and keep it untracked.
```
