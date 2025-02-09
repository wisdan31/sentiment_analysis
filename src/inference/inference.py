import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(SRC_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(model_path):
    """Load the trained model."""
    return joblib.load(model_path)

def load_test_data():
    """Load processed test data."""
    test_data_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    return pd.read_csv(test_data_path)

def run_inference():
    """Perform inference on test data and save results."""
    model_path = os.path.join(MODELS_DIR, "model_01.pkl")

    # Load model and test data
    model = load_model(model_path)
    test_data = load_test_data()

    X_test = test_data.iloc[:, :-1]  # Features
    has_sentiment = "sentiment" in test_data.columns
    y_true = test_data["sentiment"] if has_sentiment else None

    # Make predictions
    y_pred = model.predict(X_test)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame({"Predicted Sentiment": y_pred})
    results_df.to_csv(os.path.join(RESULTS_DIR, "inference_results.csv"), index=False)

    logging.info(f"Inference results saved to {os.path.join(RESULTS_DIR, 'inference_results.csv')}")

    # Log accuracy if true labels exist
    if has_sentiment:
        accuracy = accuracy_score(y_true, y_pred)
        logging.info(f"Model Accuracy on Test Data: {accuracy:.4f}")

if __name__ == "__main__":
    run_inference()
