# Sample API Requests

## Health Check

```bash
curl https://your-app-name.onrender.com/health
```

## Single Prediction

```bash
curl -X POST "https://your-app-name.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"The football team won the championship after a dramatic final match.\"}"
```

## Batch Prediction

```bash
curl -X POST "https://your-app-name.onrender.com/predict-batch" \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"The president met foreign leaders today.\", \"Stocks rose after strong earnings reports.\", \"Scientists discovered a new planet.\"]}"
```
