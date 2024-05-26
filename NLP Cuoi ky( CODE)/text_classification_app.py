import joblib
import numpy as np

def classify_text(text):
    # Load model
    model = joblib.load("text_classification_model.pkl")

    # Classify the text
    predicted_category = model.predict([text])[0]

    # Convert category number to text
    categories = {
        0: 'Tin học',
        1: 'Khoa học',
        2: 'Thể thao',
        3: 'Giải trí',
        4: 'Y tế',
        5: 'Công nghệ',
        6: 'Kinh doanh',
        7: 'Xã hội',
        8: 'Âm nhạc',
        9: 'Du lịch'
    }

    return categories[predicted_category]
