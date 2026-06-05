from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "news_classifier.joblib"
LABEL_PATH = MODEL_DIR / "label_names.json"

METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

DATASET_NAME = "ag_news"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)
