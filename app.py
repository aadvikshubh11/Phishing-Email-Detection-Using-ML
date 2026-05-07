import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)
model = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_pipeline.joblib")


def load_artifacts():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model artifacts not found. Run `python train.py` first.")
    model = joblib.load(MODEL_PATH)


load_artifacts()


def explain_prediction(text, top_n=4):
    if model is None:
        return {}

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


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email_text", "").strip()

    if not email_text:
        return render_template(
            "index.html",
            error="Please enter the email text to analyze."
        )

    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]

    result = "Phishing" if prediction == 1 else "Legitimate"
    confidence = max(probabilities) * 100
    confidence_display = f"{confidence:.2f}%"
    explanation = explain_prediction(email_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        confidence_display=confidence_display,
        email_text=email_text,
        explanation=explanation,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)

    if not data or "email_text" not in data:
        return jsonify(
            {"error": "Provide JSON with the key 'email_text'."}
        ), 400

    email_text = data["email_text"].strip()

    if not email_text:
        return jsonify(
            {"error": "email_text cannot be empty."}
        ), 400

    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0].tolist()

    explanation = explain_prediction(email_text)

    return jsonify(
        {
            "prediction": "phishing" if prediction == 1 else "legitimate",
            "probabilities": {
                "legitimate": probabilities[0],
                "phishing": probabilities[1],
            },
            "explanation": explanation,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
