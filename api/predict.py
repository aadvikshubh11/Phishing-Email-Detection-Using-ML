import json
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "phishing_pipeline.joblib"
model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model artifact not found. Run `python train.py` first.")
        model = joblib.load(MODEL_PATH)
    return model


def explain_prediction(text, top_n=4):
    model = load_model()
    features = model.named_steps["features"].transform([text])
    classifier = model.named_steps["classifier"]
    coeffs = classifier.coef_[0]
    feature_names = model.named_steps["features"].get_feature_names_out()

    contributions = features.toarray()[0] * coeffs
    sorted_idx = np.argsort(contributions)

    negative = [
        {"feature": feature_names[i], "score": float(contributions[i])}
        for i in sorted_idx[:top_n]
    ]
    positive = [
        {"feature": feature_names[i], "score": float(contributions[i])}
        for i in sorted_idx[-top_n:][::-1]
    ]

    return {"positive": positive, "negative": negative}


def json_response(body, status_code=200):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(request):
    if request.get("method") != "POST":
        return json_response({"error": "Only POST requests are allowed."}, 405)

    body = request.get("body", "")
    if isinstance(body, bytes):
        body = body.decode("utf-8")

    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return json_response({"error": "Invalid JSON payload."}, 400)

    email_text = str(data.get("email_text", "")).strip()
    if not email_text:
        return json_response({"error": "email_text cannot be empty."}, 400)

    model = load_model()
    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    explanation = explain_prediction(email_text)

    confidence = float(max(probabilities) * 100)

    return json_response(
        {
            "prediction": "phishing" if prediction == 1 else "legitimate",
            "confidence": confidence,
            "probabilities": {
                "legitimate": float(probabilities[0]),
                "phishing": float(probabilities[1]),
            },
            "explanation": explanation,
        }
    )
