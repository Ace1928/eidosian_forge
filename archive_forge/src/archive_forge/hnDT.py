import filecmp

# Files to compare
file1 = "path/to/file1"
file2 = "path/to/file2"

# Compare the files
if filecmp.cmp(file1, file2):
    print("The files are identical.")
else:
    print("The files are different.")
