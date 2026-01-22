import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def fancy_rename(old, new, rename_func, unlink_func):
    """A fancy rename, when you don't have atomic rename.

    :param old: The old path, to rename from
    :param new: The new path, to rename to
    :param rename_func: The potentially non-atomic rename function
    :param unlink_func: A way to delete the target file if the full rename
        succeeds
    """
    from .transport import NoSuchFile
    base = os.path.basename(new)
    dirname = os.path.dirname(new)
    tmp_name = 'tmp.%s.%.9f.%d.%s' % (base, time.time(), os.getpid(), rand_chars(10))
    tmp_name = pathjoin(dirname, tmp_name)
    file_existed = False
    try:
        rename_func(new, tmp_name)
    except NoSuchFile:
        pass
    except OSError as e:
        if e.errno not in (None, errno.ENOENT, errno.ENOTDIR):
            raise
    except Exception as e:
        if getattr(e, 'errno', None) is None or e.errno not in (errno.ENOENT, errno.ENOTDIR):
            raise
    else:
        file_existed = True
    success = False
    try:
        rename_func(old, new)
        success = True
    except OSError as e:
        if file_existed and e.errno in (None, errno.ENOENT) and (old.lower() == new.lower()):
            pass
        else:
            raise
    finally:
        if file_existed:
            if success:
                unlink_func(tmp_name)
            else:
                rename_func(tmp_name, new)