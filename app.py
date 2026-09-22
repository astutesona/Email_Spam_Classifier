import streamlit as st
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/spam_classifier.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# TITLE
# ============================================================

st.title(" Email Spam Classifier")

st.write(
    "Enter an email below and the machine learning model "
    "will classify it as Ham or Spam."
)


# ============================================================
# EMAIL INPUT
# ============================================================

email_text = st.text_area(
    "Enter Email Content",
    height=250,
    placeholder="Paste your email here..."
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(" Check Email"):

    if email_text.strip() == "":
        
        st.warning("Please enter an email first.")

    else:

        # Convert text into TF-IDF features
        email_vector = vectorizer.transform(
            [email_text]
        )

        # Predict
        prediction = model.predict(
            email_vector
        )[0]


        # Display result
        if prediction == "spam":

            st.error(
                "This email is classified as SPAM."
            )

        else:

            st.success(
                "This email is classified as HAM."
            )