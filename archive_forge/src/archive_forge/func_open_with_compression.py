import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def open_with_compression(filename: str, mode: str='r') -> IO:
    """
    Wrapper around builtin `open` that will guess compression of a file
    from the filename and open it for reading or writing as if it were
    a standard file.

    Implemented for ``gz``(gzip), ``bz2``(bzip2) and ``xz``(lzma).

    Supported modes are:
       * 'r', 'rt', 'w', 'wt' for text mode read and write.
       * 'rb, 'wb' for binary read and write.

    Parameters
    ==========
    filename: str
        Path to the file to open, including any extensions that indicate
        the compression used.
    mode: str
        Mode to open the file, same as for builtin ``open``, e.g 'r', 'w'.

    Returns
    =======
    fd: file
        File-like object open with the specified mode.
    """
    if mode == 'r':
        mode = 'rt'
    elif mode == 'w':
        mode = 'wt'
    elif mode == 'a':
        mode = 'at'
    root, compression = get_compression(filename)
    if compression == 'gz':
        import gzip
        return gzip.open(filename, mode=mode)
    elif compression == 'bz2':
        import bz2
        return bz2.open(filename, mode=mode)
    elif compression == 'xz':
        import lzma
        return lzma.open(filename, mode)
    else:
        return open(filename, mode)