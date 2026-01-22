import zipfile

# Files to include in the archive
files_to_archive = ["file1.txt", "file2.txt", "file3.txt"]

# Name of the ZIP archive
archive_name = "example.zip"

# Create the ZIP archive
with zipfile.ZipFile(archive_name, "w") as zip_file:
    for file in files_to_archive:
        zip_file.write(file)

print("ZIP archive created successfully.")

# Extract files from the ZIP archive
with zipfile.ZipFile(archive_name, "r") as zip_file:
    zip_file.extractall()

print("Files extracted successfully.")
