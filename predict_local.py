import json
import sys

import joblib

from config import LABEL_PATH, MODEL_PATH


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict_local.py "Your news article text here"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    prediction = int(model.predict([text])[0])
    predicted_class = label_names[prediction]

    result = {
        "text": text,
        "predicted_class": predicted_class,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        result["confidence"] = round(float(probabilities[prediction]), 4)
        result["class_probabilities"] = {
            label_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(label_names))
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
