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
def is_file_hidden_win(abs_path: str, stat_res: Optional[Any]=None) -> bool:
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
    if stat_res is None:
        try:
            stat_res = Path(abs_path).stat()
        except OSError as e:
            if e.errno == errno.ENOENT:
                return False
            raise
    try:
        if stat_res.st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN:
            return True
    except AttributeError:
        warnings.warn('hidden files are not detectable on this system, so no file will be marked as hidden.', stacklevel=2)
    return False