from __future__ import annotations
import bz2
import errno
import gzip
import io
import mmap
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING
def zopen(filename: Union[str, Path], *args, **kwargs) -> IO:
    """
    This function wraps around the bz2, gzip, lzma, xz and standard python's open
    function to deal intelligently with bzipped, gzipped or standard text
    files.

    Args:
        filename (str/Path): filename or pathlib.Path.
        *args: Standard args for python open(..). E.g., 'r' for read, 'w' for
            write.
        **kwargs: Standard kwargs for python open(..).

    Returns:
        File-like object. Supports with context.
    """
    if filename is not None and isinstance(filename, Path):
        filename = str(filename)
    _name, ext = os.path.splitext(filename)
    ext = ext.upper()
    if ext == '.BZ2':
        return bz2.open(filename, *args, **kwargs)
    if ext in ('.GZ', '.Z'):
        return gzip.open(filename, *args, **kwargs)
    if lzma is not None and ext in ('.XZ', '.LZMA'):
        return lzma.open(filename, *args, **kwargs)
    return open(filename, *args, **kwargs)