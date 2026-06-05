import json

import joblib
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import (
    CONFUSION_MATRIX_PATH,
    DATASET_NAME,
    LABEL_PATH,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    REPORT_PATH,
    TEST_SIZE,
)


def evaluate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file was not found. Run `python train.py` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading labels...")
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    _, X_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("Predicting...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
    )

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "dataset": DATASET_NAME,
        "model": "TF-IDF + Logistic Regression",
        "test_size": TEST_SIZE,
        "accuracy": accuracy,
        "labels": label_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )
    display.plot(values_format="d", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print("\nClassification Report:")
    print(report_text)


if __name__ == "__main__":
    evaluate()
