import pandas as pd


# Load dataset
df = pd.read_csv("emails.csv")

print("=" * 60)
print("EMAIL SPAM CLASSIFIER - DATA CLEANING")
print("=" * 60)

print("\nOriginal dataset:")
print(f"Rows: {len(df)}")


# Remove missing values
df = df.dropna(subset=["label", "text"])

print("\nAfter removing missing values:")
print(f"Rows: {len(df)}")


# Remove empty emails
df["text"] = df["text"].astype(str)

df = df[df["text"].str.strip() != ""]

print("\nAfter removing empty emails:")
print(f"Rows: {len(df)}")


# Remove duplicate emails
before_duplicates = len(df)

df = df.drop_duplicates(subset=["text"])

after_duplicates = len(df)

print("\nDuplicate emails removed:")
print(before_duplicates - after_duplicates)

print(f"Rows after removing duplicates: {after_duplicates}")


# Reset index
df = df.reset_index(drop=True)


# Save cleaned dataset
df.to_csv("cleaned_emails.csv", index=False)

print("\nCleaned dataset saved as:")
print("cleaned_emails.csv")


# Final class distribution
print("\nFinal class distribution:")
print(df["label"].value_counts())


print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)