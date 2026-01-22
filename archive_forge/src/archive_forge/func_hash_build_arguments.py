import collections
def hash_build_arguments(hash_value, build_arguments):
    for group in build_arguments:
        if group:
            for argument in group:
                hash_value = update_hash(hash_value, argument)
    return hash_value