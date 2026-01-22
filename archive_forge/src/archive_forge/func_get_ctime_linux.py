import os
def get_ctime_linux(filepath):
    try:
        return float(os.getxattr(filepath, b'user.loguru_crtime'))
    except OSError:
        return os.stat(filepath).st_mtime