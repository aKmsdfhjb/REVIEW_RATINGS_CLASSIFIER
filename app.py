import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"
MODEL_FILES = {
    "ensemble": MODEL_DIR / "sentiment_analysis_model.pkl",
    "svm": MODEL_DIR / "svm_sentiment_svm_model.pkl",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
origins = [
    value.strip()
    for value in os.getenv(
        "CORS_ORIGINS",
        "https://aKmsdfhjb.github.io,http://localhost:8000",
    ).split(",")
    if value.strip()
]
CORS(
    app,
    resources={
        r"/predict": {"origins": origins},
        r"/health": {"origins": origins},
    },
)

models = {}
model_errors = {}


def load_models():
    for name, path in MODEL_FILES.items():
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing model artifact: {path.name}")
            models[name] = joblib.load(path)
        except Exception as exc:
            model_errors[name] = str(exc)


@app.get("/health")
def health():
    return jsonify(
        {
            "ready": bool(models),
            "available_models": sorted(models),
            "model_errors": model_errors,
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    review = payload.get("review", "")
    model_name = payload.get("model", "ensemble")
    rating = payload.get("rating", 3)

    if not isinstance(review, str) or not review.strip():
        return jsonify({"error": "Enter a review before classifying it."}), 400
    if len(review) > 5000:
        return jsonify({"error": "Reviews must be 5,000 characters or fewer."}), 400
    if model_name not in MODEL_FILES:
        return jsonify({"error": "Choose either the ensemble or SVM model."}), 400
    if model_name not in models:
        return (
            jsonify({"error": model_errors.get(model_name, "Model unavailable.")}),
            503,
        )

    try:
        model = models[model_name]
        if model_name == "svm":
            try:
                numeric_rating = float(rating)
            except (TypeError, ValueError):
                return jsonify({"error": "Rating must be a number from 1 to 5."}), 400
            if not 1 <= numeric_rating <= 5:
                return jsonify({"error": "Rating must be between 1 and 5."}), 400
            sample = pd.DataFrame(
                [{"Review": review.strip(), "Ratings": numeric_rating}]
            )
        else:
            sample = [review.strip()]

        prediction = model.predict(sample)[0]
        result = {"prediction": str(prediction), "model": model_name}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            result["class_scores"] = {
                str(label): float(score)
                for label, score in zip(model.classes_, probabilities)
            }
            result["score_type"] = "model probability"
        else:
            result["score_type"] = "not provided by this estimator"
        return jsonify(result)
    except Exception:
        app.logger.exception("Sentiment prediction failed")
        return (
            jsonify({"error": "The selected model could not process this review."}),
            500,
        )


load_models()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
