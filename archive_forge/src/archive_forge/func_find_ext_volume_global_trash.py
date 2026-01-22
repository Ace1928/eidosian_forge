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
def find_ext_volume_global_trash(volume_root):
    trash_dir = op.join(volume_root, TOPDIR_TRASH)
    if not op.exists(trash_dir):
        return None
    mode = os.lstat(trash_dir).st_mode
    if not op.isdir(trash_dir) or op.islink(trash_dir) or (not mode & stat.S_ISVTX):
        return None
    trash_dir = op.join(trash_dir, text_type(uid).encode('ascii'))
    try:
        check_create(trash_dir)
    except OSError:
        return None
    return trash_dir