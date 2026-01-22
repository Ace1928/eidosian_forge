import errno
import os
from io import BytesIO
from typing import Set
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import bedding
def get_user_ignores():
    """Get the list of user ignored files, possibly creating it."""
    path = bedding.user_ignore_config_path()
    patterns = set(USER_DEFAULTS)
    try:
        f = open(path, 'rb')
    except OSError as e:
        err = getattr(e, 'errno', None)
        if err not in (errno.ENOENT,):
            raise
        try:
            _set_user_ignores(USER_DEFAULTS)
        except OSError as e:
            if e.errno not in (errno.EPERM, errno.ENOENT):
                raise
        return patterns
    try:
        return parse_ignore_file(f)
    finally:
        f.close()