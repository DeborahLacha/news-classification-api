import json
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    DATASET_NAME,
    LABEL_PATH,
    MAX_FEATURES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NGRAM_RANGE,
    OUTPUT_DIR,
    RANDOM_STATE,
    REPORT_PATH,
    TEST_SIZE,
)


def load_ag_news():
    dataset = load_dataset(DATASET_NAME)

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]
    label_names = dataset["train"].features["label"].names

    return texts, labels, label_names


def build_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    multi_class="auto",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    texts, labels, label_names = load_ag_news()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("Building model pipeline...")
    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "dataset": DATASET_NAME,
        "model": "TF-IDF + Logistic Regression",
        "test_size": TEST_SIZE,
        "accuracy": accuracy,
        "labels": label_names,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }

    print("Saving model...")
    joblib.dump(pipeline, MODEL_PATH)

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nTraining complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print("\nClassification Report:")
    print(report_text)


if __name__ == "__main__":
    train()
