from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = (
    "Have you ever sought treatment for a mental health issue from a mental "
    "health professional?"
)

TEXT_HEAVY_COLUMNS = [
    "Why or why not?",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
TUNING_ITERATIONS = 6
TOP_MODELS_TO_TUNE = 2

AGE_COLUMN = "What is your age?"
GENDER_COLUMN = "What is your gender?"

HIGH_MISSING_DROP_THRESHOLD = 0.40
