import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .config import Paths, TrainingConfig
from .data import load_data, split_data
from .metrics import regression_metrics
from .models import build_models
from .preprocessing import build_preprocessor


def evaluate_with_cv(pipeline, X, y, cv_folds: int) -> dict:
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(pipeline, X, y, cv=cv_folds, scoring=scoring)
    return {
        "mae": float(-scores["test_mae"].mean()),
        "rmse": float(-scores["test_rmse"].mean()),
        "r2": float(scores["test_r2"].mean()),
    }


def train_and_select(paths: Paths, config: TrainingConfig) -> dict:
    target = "Sales"
    df = load_data(paths)
    df = df.dropna()
    df = df.query("`Social Media` > 0.001 & Radio > 0.001")
    numeric_features = ["TV", "Radio", "Social Media"]
    categorical_features = ["Influencer"]

    X_train, X_test, y_train, y_test = split_data(
        df, target=target, test_size=config.test_size, random_state=config.random_state
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(random_state=config.random_state)

    results = {}
    best_name = None
    best_rmse = float("inf")
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        cv_metrics = evaluate_with_cv(pipeline, X_train, y_train, config.cv_folds)
        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)
        test_metrics = regression_metrics(y_test, test_pred)

        results[name] = {
            "cv": cv_metrics,
            "test": test_metrics,
        }

        if test_metrics["rmse"] < best_rmse:
            best_rmse = test_metrics["rmse"]
            best_name = name
            best_pipeline = pipeline

    return {
        "best_model": best_name,
        "metrics": results,
        "pipeline": best_pipeline,
    }


def save_artifacts(paths: Paths, output: dict) -> None:
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = paths.artifacts_dir / "model.joblib"
    metrics_path = paths.artifacts_dir / "metrics.json"

    joblib.dump(output["pipeline"], model_path)
    metrics_payload = {
        "best_model": output["best_model"],
        "metrics": output["metrics"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")


def main() -> None:
    paths = Paths.default()
    config = TrainingConfig()

    output = train_and_select(paths, config)
    save_artifacts(paths, output)

    print("Best model:", output["best_model"])


if __name__ == "__main__":
    main()
