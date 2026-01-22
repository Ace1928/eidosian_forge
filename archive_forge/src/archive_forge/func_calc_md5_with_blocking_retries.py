from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
from streamlit.util import HASHLIB_KWARGS
def calc_md5_with_blocking_retries(path: str, *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> str:
    """Calculate the MD5 checksum of a given path.

    For a file, this means calculating the md5 of the file's contents. For a
    directory, we concatenate the directory's path with the names of all the
    files in it and calculate the md5 of that.

    IMPORTANT: This method calls time.sleep(), which blocks execution. So you
    should only use this outside the main thread.
    """
    if allow_nonexistent and (not os.path.exists(path)):
        content = path.encode('UTF-8')
    elif os.path.isdir(path):
        glob_pattern = glob_pattern or '*'
        content = _stable_dir_identifier(path, glob_pattern).encode('UTF-8')
    else:
        content = _get_file_content_with_blocking_retries(path)
    md5 = hashlib.md5(**HASHLIB_KWARGS)
    md5.update(content)
    return md5.hexdigest()