import os
import sys
import stat
import select
import time
import errno
def is_executable_file(path):
    """Checks that path is an executable regular file, or a symlink towards one.

    This is roughly ``os.path isfile(path) and os.access(path, os.X_OK)``.
    """
    fpath = os.path.realpath(path)
    if not os.path.isfile(fpath):
        return False
    mode = os.stat(fpath).st_mode
    if sys.platform.startswith('sunos') and os.getuid() == 0:
        return bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    return os.access(fpath, os.X_OK)