import os
import json
from datetime import datetime
from joblib import dump, load

from scripts.config import appconfig

# Get directories from config
MODEL_DIR = appconfig['Directories']['model_dir']
METADATA_DIR = appconfig['Directories']['metadata_dir']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def get_next_version(model_name):
    """Finds the next version number for the given model name."""
    versions = [0]
    for file in os.listdir(METADATA_DIR):
        if file.startswith(model_name):
            try:
                version = int(file.split('_v')[-1].split('.')[0])
                versions.append(version)
            except:
                continue
    return max(versions) + 1

def register(model, features, metadata):
    """Saves model, features, and metadata as a new version."""
    version = get_next_version(metadata['name'])
    metadata['version'] = version
    metadata['registration_date'] = datetime.now().isoformat()

    # File names
    model_file = f"{metadata['name']}_model_v{version}.joblib"
    features_file = f"{metadata['name']}_features_v{version}.joblib"
    metadata_file = f"{metadata['name']}_v{version}.json"

    # Save model and features
    dump(model, os.path.join(MODEL_DIR, model_file))
    dump(features, os.path.join(MODEL_DIR, features_file))

    # Add file names to metadata and save
    metadata['model'] = model_file
    metadata['features'] = features_file
    with open(os.path.join(METADATA_DIR, metadata_file), "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_file

def retrieve(model_name, version=None):
    """Loads a model and its features from registry."""
    if version is None:
        version = get_next_version(model_name) - 1

    metadata_file = os.path.join(METADATA_DIR, f"{model_name}_v{version}.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    model = load(os.path.join(MODEL_DIR, metadata["model"]))
    features = load(os.path.join(MODEL_DIR, metadata["features"]))
    return model, features