"""Project paths shared by all notebooks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
STRATEGY_DIR = ARTIFACTS_DIR / "strategy"
NOTEBOOKS_DIR = ROOT / "notebooks"

TRAIN_CSV = DATA_DIR / "fraud_credit_train.csv"
TEST_CSV = DATA_DIR / "fraud_credit_test.csv"
RAW_CSV = DATA_DIR / "Fraudulent_E-Commerce_Transaction_Data.csv"

XGB_MODEL_PATH = MODELS_DIR / "best_xgb_model.joblib"
LOGISTIC_MODEL_PATH = MODELS_DIR / "best_logistic_model.joblib"
XGB_GRID_PATH = MODELS_DIR / "xgb_grid_search.joblib"
LOGISTIC_GRID_PATH = MODELS_DIR / "logistic_grid_search.joblib"

FROZEN_STRATEGY_PATH = STRATEGY_DIR / "frozen_strategy.json"
FP_SENSITIVITY_PATH = ARTIFACTS_DIR / "fp_cost_sensitivity.csv"
