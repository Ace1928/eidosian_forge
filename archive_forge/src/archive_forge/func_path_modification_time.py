from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
from streamlit.util import HASHLIB_KWARGS
def path_modification_time(path: str, allow_nonexistent: bool=False) -> float:
    """Return the modification time of a path (file or directory).

    If allow_nonexistent is True and the path does not exist, we return 0.0 to
    guarantee that any file/dir later created at the path has a later
    modification time than the last time returned by this function for that
    path.

    If allow_nonexistent is False and no file/dir exists at the path, a
    FileNotFoundError is raised (by os.stat).

    For any path that does correspond to an existing file/dir, we return its
    modification time.
    """
    if allow_nonexistent and (not os.path.exists(path)):
        return 0.0
    return os.stat(path).st_mtime