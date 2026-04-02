def policy_gate(nlu_result, stt_confidence):
    intent = nlu_result.get("intent", "UNKNOWN")
    nlu_confidence = nlu_result.get("confidence", 0.0)

    if stt_confidence < 0.6 or nlu_confidence < 0.5:
        return "confirm"

    if intent == "DEVICE_CONTROL":
        return "allow"
    elif intent == "TIMER_SET":
        return "allow"
    elif intent == "WEATHER_QUERY":
        return "allow"
    elif intent == "GREETING":
        return "allow"
    elif intent == "UNKNOWN":
        return "confirm"
    else:
        return "deny"


def generate_response(nlu_result, policy_action):
    intent = nlu_result.get("intent", "UNKNOWN")
    slots = nlu_result.get("slots", {})

    if policy_action == "deny":
        return "I can't do that."

    if policy_action == "confirm":
        return "I didn't catch that — can you repeat?"

    if intent == "DEVICE_CONTROL":
        action = slots.get("action", "")
        device = slots.get("device", "")

        if action and device:
            pretty_action = action.replace("_", " ")
            return f"{pretty_action.capitalize()} the {device}."

        return "Controlling device."

    elif intent == "TIMER_SET":
        duration = slots.get("duration")

        if duration:
            return f"Setting a timer for {duration}."

        return "Setting timer."

    elif intent == "WEATHER_QUERY":
        location = slots.get("location")

        if location:
            return f"Checking the weather in {location}."

        return "Checking the weather."

    elif intent == "GREETING":
        return "Hello! How can I help you?"

    elif intent == "UNKNOWN":
        return "Sorry, I didn't understand that."

    return "I don't know how to respond."


def process_nlu_result(nlu_result, stt_confidence):
    policy_action = policy_gate(nlu_result, stt_confidence)
    response_text = generate_response(nlu_result, policy_action)

    return {
        "intent": nlu_result.get("intent"),
        "slots": nlu_result.get("slots"),
        "confidence": nlu_result.get("confidence"),
        "policy_action": policy_action,
        "response_text": response_text,
    }
