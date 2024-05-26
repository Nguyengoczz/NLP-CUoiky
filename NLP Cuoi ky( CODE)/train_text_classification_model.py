import joblib
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from load_my_dataset_data import load_my_dataset_data  # Import hàm load dữ liệu

# Load dữ liệu
X, y, target_names = load_my_dataset_data()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng pipeline cho mô hình
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Lưu mô hình
joblib.dump(pipeline, "text_classification_model.pkl")
print("Model saved as text_classification_model.pkl")
