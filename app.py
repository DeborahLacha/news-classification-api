import json
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import LABEL_PATH, MODEL_PATH


app = FastAPI(
    title="News Article Text Classification API",
    description="A prototype API for classifying news articles into multiple classes.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="News article text to classify.",
        example="Apple announced a new AI chip for its upcoming devices.",
    )


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of news article texts to classify.",
        example=[
            "The government announced a new foreign policy today.",
            "The football team won the championship.",
        ],
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]


class BatchPredictionItem(BaseModel):
    text: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]


model = None
label_names = None


def load_artifacts():
    global model, label_names

    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        raise RuntimeError(
            "Model artifacts were not found. Run `python train.py` before starting the API."
        )

    model = joblib.load(MODEL_PATH)

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)


@app.on_event("startup")
def startup_event():
    load_artifacts()


def predict_one(text: str):
    if model is None or label_names is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    predicted_index = int(model.predict([text])[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
    else:
        probabilities = np.zeros(len(label_names))
        probabilities[predicted_index] = 1.0

    probability_map = {
        label_names[i]: round(float(probabilities[i]), 4)
        for i in range(len(label_names))
    }

    confidence = round(float(probabilities[predicted_index]), 4)

    return {
        "text": text,
        "predicted_class": label_names[predicted_index],
        "confidence": confidence,
        "class_probabilities": probability_map,
    }


@app.get("/")
def home():
    return {
        "message": "News Article Classification API is running.",
        "available_endpoints": {
            "health": "/health",
            "single_prediction": "/predict",
            "batch_prediction": "/predict-batch",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": label_names is not None,
    }


@app.get("/labels")
def get_labels():
    if label_names is None:
        raise HTTPException(status_code=500, detail="Labels are not loaded.")
    return {"labels": label_names}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return predict_one(request.text)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if len(request.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum allowed is 50 texts.",
        )

    predictions = [predict_one(text) for text in request.texts]
    return {"predictions": predictions}
