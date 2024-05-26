import joblib

def classify_text(text):
    # Load model
    model = joblib.load("text_classification_model.pkl")

    # Classify text
    category = model.predict([text])[0]
    return category
