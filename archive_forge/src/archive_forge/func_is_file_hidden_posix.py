from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def is_file_hidden_posix(abs_path: str, stat_res: Optional[Any]=None) -> bool:
    """Is a file hidden?

    This only checks the file itself; it should be called in combination with
    checking the directory containing the file.

    Use is_hidden() instead to check the file and its parent directories.

    Parameters
    ----------
    abs_path : unicode
        The absolute path to check.
    stat_res : os.stat_result, optional
        The result of calling stat() on abs_path. If not passed, this function
        will call stat() internally.
    """
    if Path(abs_path).name.startswith('.'):
        return True
    if stat_res is None or stat.S_ISLNK(stat_res.st_mode):
        try:
            stat_res = Path(abs_path).stat()
        except OSError as e:
            if e.errno == errno.ENOENT:
                return False
            raise
    if stat.S_ISDIR(stat_res.st_mode):
        if not os.access(abs_path, os.X_OK | os.R_OK):
            return True
    if getattr(stat_res, 'st_flags', 0) & UF_HIDDEN:
        return True
    return False