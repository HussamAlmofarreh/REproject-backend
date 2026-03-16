def recognize_intent(transcript: str) -> dict:
    transcript = transcript.lower().strip()

    intent = "UNKNOWN"
    slots = {}
    confidence = 0.0

    # Greeting
    if transcript in ["hi", "hello", "hey"]:
        intent = "GREETING"
        confidence = 0.95

    # Weather
    elif "weather" in transcript:
        intent = "WEATHER_QUERY"
        confidence = 0.9

    # Timer
    elif "timer" in transcript:
        intent = "TIMER_SET"
        confidence = 0.85
        slots["duration"] = "unknown"

    # Device control - turn on
    elif "turn on" in transcript or "switch on" in transcript:
        intent = "DEVICE_CONTROL"
        confidence = 0.9
        slots["action"] = "turn_on"

        if "light" in transcript:
            slots["device"] = "light"
        elif "fan" in transcript:
            slots["device"] = "fan"

        if "bedroom" in transcript:
            slots["location"] = "bedroom"
        elif "kitchen" in transcript:
            slots["location"] = "kitchen"

    # Device control - turn off
    elif "turn off" in transcript or "switch off" in transcript:
        intent = "DEVICE_CONTROL"
        confidence = 0.9
        slots["action"] = "turn_off"

        if "light" in transcript:
            slots["device"] = "light"
        elif "fan" in transcript:
            slots["device"] = "fan"

        if "bedroom" in transcript:
            slots["location"] = "bedroom"
        elif "kitchen" in transcript:
            slots["location"] = "kitchen"

    return {
        "intent": intent,
        "slots": slots,
        "confidence": confidence
    }
