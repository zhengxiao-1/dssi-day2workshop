# HDB Resale Price Predictor

A Streamlit web app to predict HDB resale prices in Singapore using a trained machine learning model.

## Features

- Select region, town, flat type, storey level, and floor area
- Predict resale price with a trained model
- Simple UI, fast prediction, no data entry required

## Project Structure
hdb/
├── scripts/
│ ├── training.py
│ ├── inference.py
│ ├── data_processor.py
│ ├── model_registry.py
│ └── config/
│ └── config.py
├── models/
├── metadata/
├── data/
│ └── Combined & Processed.csv
├── hdb_app.py
├── requirements.txt
└── README.md


## Getting Started

1. **Install dependencies**  
   *(recommended: use a virtual environment)*

   ```bash
   pip install -r requirements.txt

2. **Train the model** 
    ```bash
    python -m scripts.training

3. **Run the Streamlit app locally**
    ```bash
    streamlit run hdb_app.py