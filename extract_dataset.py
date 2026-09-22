import os
import tarfile
import csv
from email import policy
from email.parser import BytesParser


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

DATASET_DIR = "dataset"
OUTPUT_FILE = "emails.csv"


# ============================================================
# 2. CHECK DATASET FOLDER
# ============================================================

if not os.path.exists(DATASET_DIR):
    print("ERROR: 'dataset' folder was not found.")
    print("Make sure this script is inside Email_Spam_Classifier.")
    exit()


# ============================================================
# 3. FIND TAR.BZ2 FILES
# ============================================================

archive_files = [
    file
    for file in os.listdir(DATASET_DIR)
    if file.endswith(".tar.bz2")
    and "(1)" not in file
]


# ============================================================
# 4. DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("        SPAMASSASSIN EMAIL DATASET PROCESSOR")
print("=" * 60)

print(f"\nDataset folder: {os.path.abspath(DATASET_DIR)}")
print(f"Found {len(archive_files)} archive files.\n")

if len(archive_files) == 0:
    print("ERROR: No .tar.bz2 files found.")
    print("Please check that your files are inside the dataset folder.")
    exit()


print("Files that will be processed:")

for file in archive_files:
    print(" -", file)

print()


# ============================================================
# 5. FUNCTION TO READ EMAIL
# ============================================================

def read_email(file_object):

    try:

        # Parse the email
        message = BytesParser(
            policy=policy.default
        ).parse(file_object)

        # Try to get plain text or HTML body
        body = message.get_body(
            preferencelist=("plain", "html")
        )

        if body is not None:

            text = body.get_content()

            if text:
                return str(text)

        # Fallback method
        payload = message.get_payload()

        if isinstance(payload, str):
            return payload

        return ""

    except Exception:

        return ""


# ============================================================
# 6. STORE ALL EMAILS
# ============================================================

emails = []


# ============================================================
# 7. PROCESS EACH ARCHIVE
# ============================================================

for archive_name in archive_files:

    archive_path = os.path.join(
        DATASET_DIR,
        archive_name
    )

    # --------------------------------------------
    # Determine whether email is HAM or SPAM
    # --------------------------------------------

    if "spam" in archive_name.lower():

        label = "spam"

    elif "ham" in archive_name.lower():

        label = "ham"

    else:

        print(f"Skipping unknown archive: {archive_name}")
        continue


    print("-" * 60)
    print(f"Processing : {archive_name}")
    print(f"Label      : {label}")


    try:

        # Open .tar.bz2 archive
        with tarfile.open(
            archive_path,
            mode="r:bz2"
        ) as archive:

            members = archive.getmembers()

            count = 0

            # --------------------------------------------
            # Read every file inside the archive
            # --------------------------------------------

            for member in members:

                # Ignore folders
                if not member.isfile():
                    continue

                # Ignore unwanted system files
                if member.name.startswith("__"):
                    continue

                try:

                    # Extract file from archive into memory
                    file_object = archive.extractfile(member)

                    if file_object is None:
                        continue


                    # Read email
                    text = read_email(file_object)


                    # Ignore empty emails
                    if not text.strip():
                        continue


                    # Store email
                    emails.append({
                        "label": label,
                        "text": text.strip()
                    })


                    count += 1


                except Exception:

                    # If one email causes an error,
                    # continue with the next email
                    continue


            print(f"Emails collected: {count}")


    except Exception as e:

        print(f"ERROR processing {archive_name}")
        print(f"Reason: {e}")


# ============================================================
# 8. CHECK WHETHER EMAILS WERE FOUND
# ============================================================

if len(emails) == 0:

    print("\n")
    print("=" * 60)
    print("ERROR: NO EMAILS WERE FOUND")
    print("=" * 60)

    print("\nPossible reasons:")
    print("1. The archive files may be corrupted.")
    print("2. The archives may have an unexpected structure.")
    print("3. The files may not contain email messages.")

    exit()


# ============================================================
# 9. CREATE emails.csv
# ============================================================

print("\n")
print("=" * 60)
print("CREATING emails.csv")
print("=" * 60)


with open(
    OUTPUT_FILE,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["label", "text"]
    )

    # Write column names
    writer.writeheader()

    # Write all emails
    writer.writerows(emails)


# ============================================================
# 10. CALCULATE DATASET STATISTICS
# ============================================================

spam_count = sum(
    1
    for email in emails
    if email["label"] == "spam"
)


ham_count = sum(
    1
    for email in emails
    if email["label"] == "ham"
)


total_count = len(emails)


# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("       DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Total emails : {total_count}")
print(f"Ham emails   : {ham_count}")
print(f"Spam emails  : {spam_count}")

print("\nCSV file:")
print(os.path.abspath(OUTPUT_FILE))

print("=" * 60)
print("Step 1 completed!")
print("=" * 60)