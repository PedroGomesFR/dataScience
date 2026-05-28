from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.config import Paths
from src.data import load_data
from src.preprocessing import build_preprocessor


class PredictionRequest(BaseModel):
    tv: float = Field(..., ge=0)
    radio: float = Field(..., ge=0)
    social_media: float = Field(..., ge=0)
    influencer: str


class PredictionResponse(BaseModel):
    predicted_sales: float


class ModelInfo(BaseModel):
    model_path: str
    model_type: Optional[str]
    estimator_name: Optional[str] = None


app = FastAPI(title="Marketing ROI Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"
LINEAR_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "linear_regression.joblib"
model = None


def build_linear_regression_model() -> Pipeline:
    paths = Paths.default()
    df = load_data(paths)
    df = df.dropna()
    df = df.query("`Social Media` > 0.001 & Radio > 0.001")

    numeric_features = ["TV", "Radio", "Social Media"]
    categorical_features = ["Influencer"]
    X = df.drop(columns=["Sales"])
    y = df["Sales"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@app.on_event("startup")
def load_model() -> None:
    global model
    model = None

    if LINEAR_MODEL_PATH.exists():
        model = joblib.load(LINEAR_MODEL_PATH)
        return

    try:
        model = build_linear_regression_model()
        LINEAR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, LINEAR_MODEL_PATH)
    except Exception:
        model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_type = getattr(model, "__class__", None)
    estimator_name = None
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator_name = model.named_steps["model"].__class__.__name__

    model_path = LINEAR_MODEL_PATH if LINEAR_MODEL_PATH.exists() else MODEL_PATH

    return ModelInfo(
        model_path=str(model_path),
        model_type=str(model_type),
        estimator_name=estimator_name,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = pd.DataFrame(
        [
            {
                "TV": payload.tv,
                "Radio": payload.radio,
                "Social Media": payload.social_media,
                "Influencer": payload.influencer,
            }
        ]
    )

    try:
        prediction = float(model.predict(data)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(predicted_sales=prediction)
