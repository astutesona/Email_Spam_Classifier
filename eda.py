import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("emails.csv")

print("=" * 60)
print("EMAIL SPAM CLASSIFIER - DATASET ANALYSIS")
print("=" * 60)


# Basic information
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# Data types
print("\nData types:")
print(df.dtypes)


# Missing values
print("\nMissing values:")
print(df.isnull().sum())


# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())


# Class distribution
print("\nClass distribution:")
print(df["label"].value_counts())


# Class percentage
print("\nClass distribution percentage:")
print(
    df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# Ham vs Spam graph
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="label"
)

plt.title("Ham vs Spam Email Distribution")
plt.xlabel("Email Type")
plt.ylabel("Number of Emails")

plt.tight_layout()
plt.show()


# Email length
df["text_length"] = df["text"].astype(str).apply(len)

print("\nEmail text length statistics:")
print(df["text_length"].describe())


# Email length graph
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="text_length",
    hue="label",
    bins=50,
    kde=True
)

plt.title("Email Text Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Emails")

plt.tight_layout()
plt.show()


# Average email length
print("\nAverage email length by class:")

print(
    df.groupby("label")["text_length"].mean()
)


print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print(f"Total emails : {len(df)}")
print(f"Ham emails   : {(df['label'] == 'ham').sum()}")
print(f"Spam emails  : {(df['label'] == 'spam').sum()}")