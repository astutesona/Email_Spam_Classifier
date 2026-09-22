# Email Spam Classifier

A machine learning project that classifies email content as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and machine learning.

The project covers the complete workflow from dataset preprocessing and exploratory data analysis to model comparison, evaluation, model saving, and deployment with Streamlit.

## Live Demo

**Try the application:**
https://emailspamclassifier-4jahukpdz6d3mqhpxdhpzj.streamlit.app/

## Project Overview

The goal of this project was to build an end-to-end text classification system and understand how machine learning can be applied to a real-world email classification problem.

The project includes:

* Data extraction and preprocessing
* Exploratory Data Analysis (EDA)
* Duplicate removal and data cleaning
* Text feature extraction using TF-IDF
* Training multiple machine learning models
* Model performance comparison
* Model evaluation using standard classification metrics
* Saving the trained model and vectorizer
* Building a Streamlit web application
* Deployment using Streamlit Community Cloud

## Dataset

The project uses the **SpamAssassin Public Corpus**, containing legitimate (ham) and spam emails.

### Dataset Statistics

| Category     | Number of Emails |
| ------------ | ---------------: |
| Total Emails |           10,693 |
| Ham          |            6,954 |
| Spam         |            3,739 |

After removing duplicate emails, the dataset contained **6,028 unique emails**.

Final cleaned dataset:

| Category | Number of Emails |
| -------- | ---------------: |
| Ham      |            4,308 |
| Spam     |            1,720 |

## Machine Learning Pipeline

```text
Raw Email Dataset
       ↓
Data Extraction
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Train / Test Split
       ↓
TF-IDF Feature Extraction
       ↓
Model Training
       ↓
Model Comparison
       ↓
Model Evaluation
       ↓
Save Model + Vectorizer
       ↓
Streamlit Application
       ↓
Deployment
```

## Feature Extraction

Email text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF helps the model identify words that are important within an email while reducing the importance of very common words.

The vectorizer was configured with a maximum of **5,000 features**.

## Models Compared

Three machine learning algorithms were evaluated:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (Linear SVM)

### Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Naive Bayes         |   94.53% |    87.77% | 93.90% |   90.73% |
| Logistic Regression |   97.43% |    98.75% | 92.15% |   95.34% |
| Linear SVM          |   98.92% |    98.82% | 97.38% |   98.10% |

On the test set used in this project, **Linear SVM achieved the highest performance across the reported metrics** and was selected as the final model.

## Application

The trained model and TF-IDF vectorizer are integrated into a Streamlit application.

Users can:

1. Enter or paste email content.
2. Submit the email for classification.
3. Receive a prediction as either:

   * **Ham**
   * **Spam**

## Project Structure

```text
Email_Spam_Classifier/
│
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── extract_dataset.py
├── eda.py
├── clean_data.py
├── train_model.py
├── compare_models.py
├── save_model.py
├── requirements.txt
└── .gitignore
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF
* Linear SVM
* Joblib
* Streamlit
* Matplotlib

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/astutesona/Email_Spam_Classifier.git
cd Email_Spam_Classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Key Learnings

This project helped me understand the complete machine learning development workflow:

* Working with raw text data
* Cleaning and preparing real-world datasets
* Performing exploratory data analysis
* Converting text into numerical features
* Comparing different machine learning algorithms
* Understanding classification metrics
* Saving and reusing trained models
* Building a simple ML web application
* Deploying an ML project for public use

## Future Improvements

Some possible improvements include:

* Testing additional NLP techniques
* Hyperparameter tuning
* Using larger and more diverse email datasets
* Handling multilingual emails
* Adding confidence/probability information where appropriate
* Improving the user interface
* Adding more advanced NLP models for comparison

## Limitations

The model's performance depends on the dataset used for training. Real-world emails can contain new patterns, domains, languages, and writing styles that may not be represented in the training data.

Therefore, the reported metrics should be considered specific to the dataset and test split used in this project.

## Author

**Sonali Kumari**
Final-Year MCA Student

This project was built as part of my learning journey in Machine Learning, NLP, and Software Development.

## Live Application

**https://emailspamclassifier-4jahukpdz6d3mqhpxdhpzj.streamlit.app/**
