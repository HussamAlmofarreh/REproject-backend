\# API Spec v1



Endpoint:

POST /transcribe



Input:

WAV file



Required:

16kHz mono



Pipeline:

validate

preprocess

stt

confidence

response



Response:



{

&nbsp; "api\_version": "1.0",

&nbsp; "transcript": "",

&nbsp; "confidence": 0.0,

&nbsp; "preprocess\_ms": 0

}

