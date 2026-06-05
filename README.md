# News Article Text Classification API

This is a full implementation of a multi-class news article classifier.

It includes:

- Dataset loading using AG News
- Text preprocessing with TF-IDF
- Model training
- Model evaluation
- Confusion matrix generation
- Saved model artifacts
- FastAPI prediction endpoint
- Render deployment configuration
- Docker support

## Classes

The AG News dataset contains 4 classes:

1. World
2. Sports
3. Business
4. Sci/Tech

## Project Structure

```text
news_text_classifier/
├── app.py
├── train.py
├── evaluate.py
├── predict_local.py
├── config.py
├── requirements.txt
├── render.yaml
├── Dockerfile
├── model/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
└── README.md
```

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Train the Model

```bash
python train.py
```

This saves:

```text
model/news_classifier.joblib
model/label_names.json
outputs/evaluation_metrics.json
outputs/classification_report.txt
outputs/confusion_matrix.png
```

## 3. Evaluate the Model

```bash
python evaluate.py
```

## 4. Test Prediction Locally

```bash
python predict_local.py "The football team won the championship after a dramatic final match."
```

## 5. Run FastAPI Locally

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 6. Prediction Endpoint

### Endpoint

```http
POST /predict
```

### Request

```json
{
  "text": "Apple announced a new AI chip for its upcoming devices."
}
```

### Response

```json
{
  "text": "Apple announced a new AI chip for its upcoming devices.",
  "predicted_class": "Sci/Tech",
  "confidence": 0.91,
  "class_probabilities": {
    "World": 0.01,
    "Sports": 0.01,
    "Business": 0.07,
    "Sci/Tech": 0.91
  }
}
```

## 7. Batch Prediction Endpoint

### Endpoint

```http
POST /predict-batch
```

### Request

```json
{
  "texts": [
    "The president met foreign leaders today.",
    "The basketball team won the final game.",
    "Stocks rose after strong earnings reports."
  ]
}
```

## 8. Deploy on Render

Push this project to GitHub.

Then create a Render Web Service using:

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The included `render.yaml` can also be used for blueprint deployment.

Important: On first deployment, the API will train the model automatically if the model files do not exist. This may take a little time during startup. For production, train locally and commit/upload the model artifacts or use persistent storage.

## 9. Docker Run

```bash
docker build -t news-classifier-api .
docker run -p 8000:8000 news-classifier-api
```

Then open:

```text
http://localhost:8000/docs
```
