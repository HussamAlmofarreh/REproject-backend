import argparse  # allows command line arguments
import json
import mimetypes
import os
import sys
import requests  # sends HTTP request to the backend server


# checking if the file they gave u ends with .wav
def is_wav(path: str) -> bool:
    return path.lower().endswith(".wav")


def main():

    parser = argparse.ArgumentParser(
        description="Client script: send a WAV file to /stt and print JSON response."
    )

    # REQUIRED arguenet: it's the path to the WAV file
    parser.add_argument("wav_path", help="Path to a .wav audio file")

    # OPTIONAL: backend server endpoint URL
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/stt",
        help="Backend endpoint URL"
    )

    # pretty format
    parser.add_argument("--pretty", action="store_true")

    # Jupyter-safe
    args, _ = parser.parse_known_args()

    # Store the WAV file path provided by the user
    wav_path = args.wav_path

    # check that the file actually exists
    if not os.path.isfile(wav_path):
        print(f"ERROR: File not found: {wav_path}")
        sys.exit(2)

    # check tht the file is a .wav AUDIO FILE
    if not is_wav(wav_path):
        print("ERROR: Input must be a .wav file")
        sys.exit(2)

    # guessing the file type SHOULD BE audio/wav
    guessed_type, _ = mimetypes.guess_type(wav_path)
    content_type = guessed_type or "audio/wav"

    # open the WAV file in binary mode AKA rb
    with open(wav_path, "rb") as f:
        files = {

            # file name must math wht backend expects
            "audio": (os.path.basename(wav_path), f, content_type)
        }

        # sends POST request to backend /stt endpoint
        response = requests.post(args.url, files=files)

    # print http status code ( 200 means SUCCESS)
    print("HTTP Status:", response.status_code)

    # Try to print JSON response returned by server
    try:
        print(json.dumps(response.json(), indent=2 if args.pretty else None))

    # If server response is not JSON, print raw text instead
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
