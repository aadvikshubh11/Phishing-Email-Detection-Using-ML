from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "phishing_pipeline.joblib"
model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model artifact not found.")
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
        {
            "feature": feature_names[i],
            "score": float(contributions[i])
        }
        for i in sorted_idx[:top_n]
    ]

    positive = [
        {
            "feature": feature_names[i],
            "score": float(contributions[i])
        }
        for i in sorted_idx[-top_n:][::-1]
    ]

    return {
        "positive": positive,
        "negative": negative
    }


@app.route("/")
def home():
    return "Phishing Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "email_text" not in data:
        return jsonify({
            "error": "email_text is required"
        }), 400

    email_text = data["email_text"]

    model = load_model()

    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]

    explanation = explain_prediction(email_text)

    confidence = float(max(probabilities) * 100)

    return jsonify({
        "prediction": "phishing" if prediction == 1 else "legitimate",
        "confidence": confidence,
        "probabilities": {
            "legitimate": float(probabilities[0]),
            "phishing": float(probabilities[1])
        },
        "explanation": explanation
    })


if __name__ == "__main__":
    app.run()