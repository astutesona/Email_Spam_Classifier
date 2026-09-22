import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv("cleaned_emails.csv")

X = df["text"]
y = df["label"]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)


# ============================================================
# 4. TRAIN FINAL MODEL
# ============================================================

model = LinearSVC()

model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 5. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/spam_classifier.pkl"
)


# ============================================================
# 6. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


print("=" * 60)
print("MODEL SAVING COMPLETED")
print("=" * 60)

print("\nSaved files:")

print("models/spam_classifier.pkl")
print("models/tfidf_vectorizer.pkl")