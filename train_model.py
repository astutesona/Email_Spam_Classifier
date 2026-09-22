import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv("cleaned_emails.csv")

print("=" * 60)
print("EMAIL SPAM CLASSIFIER - MODEL TRAINING")
print("=" * 60)

print("\nDataset:")
print(df.shape)


# ============================================================
# 2. INPUT AND TARGET
# ============================================================

X = df["text"]
y = df["label"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF conversion completed.")

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape :", X_test_tfidf.shape)


# ============================================================
# 5. TRAIN NAIVE BAYES MODEL
# ============================================================

model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)

print("\nModel training completed.")


# ============================================================
# 6. PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(f"{accuracy:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)