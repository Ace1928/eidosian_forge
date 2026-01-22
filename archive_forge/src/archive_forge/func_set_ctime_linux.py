import os
def set_ctime_linux(filepath, timestamp):
    try:
        os.setxattr(filepath, b'user.loguru_crtime', str(timestamp).encode('ascii'))
    except OSError:
        pass