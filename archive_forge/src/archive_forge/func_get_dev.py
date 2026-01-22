from __future__ import unicode_literals
import errno
import sys
import os
import shutil
import os.path as op
from datetime import datetime
import stat
from send2trash.compat import text_type, environb
from send2trash.util import preprocess_paths
from send2trash.exceptions import TrashPermissionError
def get_dev(path):
    return os.lstat(path).st_dev