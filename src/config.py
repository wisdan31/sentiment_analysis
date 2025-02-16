from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = Path(SRC_DIR) / "data" / "raw"
PROCESSED_DATA_DIR = Path(SRC_DIR) / "data" / "processed"
MODELS_DIR = Path(PROJECT_ROOT) / "outputs" / "models"
RESULTS_DIR = Path(PROJECT_ROOT) / "outputs" / "results"