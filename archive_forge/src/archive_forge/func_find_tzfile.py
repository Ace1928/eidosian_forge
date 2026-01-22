import os
import sysconfig
def find_tzfile(key):
    """Retrieve the path to a TZif file from a key."""
    _validate_tzfile_path(key)
    for search_path in TZPATH:
        filepath = os.path.join(search_path, key)
        if os.path.isfile(filepath):
            return filepath
    return None