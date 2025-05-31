import pandas as pd
import os
from scripts.config import appconfig

def load_training_data(file_path=None):
    """
    Loads the training data from a specified CSV file.
    If file_path is None, it defaults to the path in appconfig.

    Args:
        file_path (str, optional): The path to the CSV file containing the training data.
                                   Defaults to appconfig['Paths']['data_path'].

    Returns:
        pd.DataFrame: A DataFrame containing the loaded training data.
                      Returns None if the file is not found or an error occurs.
    """
    if file_path is None:
        file_path = appconfig['Paths']['data_path']

    if not os.path.exists(file_path):
        print(f"Error: Training data file not found at {file_path}. Please ensure the file exists.")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"Training data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading training data from {file_path}: {e}")
        return None
