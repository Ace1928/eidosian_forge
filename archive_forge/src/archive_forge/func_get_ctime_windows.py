import os
def get_ctime_windows(filepath):
    return os.stat(filepath).st_ctime