import os
def get_ctime_fallback(filepath):
    return os.stat(filepath).st_mtime