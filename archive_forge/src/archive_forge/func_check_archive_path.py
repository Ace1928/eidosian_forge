import ctypes
import hashlib
import os
import pathlib
import platform
import sys
import time as _time
import zlib
from datetime import datetime, timedelta, timezone, tzinfo
from typing import BinaryIO, List, Optional, Union
import py7zr.win32compat
from py7zr import Bad7zFile
from py7zr.win32compat import is_windows_native_python, is_windows_unc_path
def check_archive_path(arcname: str) -> bool:
    """
    Check arcname argument is valid for archive.
    It should not be absolute, if so it returns False.
    It should not be evil traversal attack path.
    Otherwise, returns True.
    """
    if pathlib.PurePath(arcname).is_absolute():
        return False
    if sys.platform == 'win32':
        path = pathlib.Path('C:/foo/boo/fuga/hoge/a90sufoiasj09/dafj08sajfa/')
    else:
        path = pathlib.Path('/foo/boo/fuga/hoge/a90sufoiasj09/dafj08sajfa/')
    return is_path_valid(path.joinpath(arcname), path)