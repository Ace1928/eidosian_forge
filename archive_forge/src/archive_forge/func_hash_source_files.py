import collections
def hash_source_files(hash_value, source_files):
    for filename in source_files:
        with open(filename) as file:
            hash_value = update_hash(hash_value, file.read())
    return hash_value