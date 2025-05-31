import logging
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from scripts.data_processor import load_training_data
from scripts.model_registry import register
from scripts.config import appconfig

logging.basicConfig(level=logging.INFO)

# Read settings from config
features = appconfig["Model"]["features"].split(",")
categorical_features = appconfig["Model"]["categorical_features"].split(",")
numerical_features = appconfig["Model"]["numerical_features"].split(",")
label = appconfig["Model"]["label"]

def run(data_path):
    """
    Main script to perform model training and registration.
    Parameters:
        data_path (str): Path to CSV training dataset
    Returns:
        None
    """
    logging.info("Loading and processing training data...")
    df = load_training_data(data_path)

    # Validate required columns
    required_columns = features + [label]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns in data: {missing_columns}")
        return

    X = df[features]
    y = df[label]

    # Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    # Define pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Split and train
    logging.info("Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    # Register model
    metadata = {
        "name": appconfig["Model"]["name"],
        "description": "Linear regression model for HDB resale price prediction"
    }

    try:
        register(pipeline, features, metadata)
        logging.info("Model registered successfully.")
    except Exception as e:
        logging.error(f"Model registration failed: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=appconfig['Paths']['data_path'],
        help="Path to training data. Defaults to value in config.ini"
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.data_path}")
    run(args.data_path)

