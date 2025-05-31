def predict_price(model, features, input_data):
    """
    Makes a prediction using the provided model and input data.

    Args:
        model (object): The trained model pipeline.
        features (list): Ordered list of features the model expects.
        input_data (pd.DataFrame): User input.

    Returns:
        float: Predicted resale price or None on error.
    """
    if model is None or features is None:
        print("Error: Model or features not loaded.")
        return None
    try:
        aligned_input = input_data[features]  # align feature order
        prediction = model.predict(aligned_input)
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

