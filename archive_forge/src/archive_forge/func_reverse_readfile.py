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
def reverse_readfile(filename: Union[str, Path]) -> Generator[str, str, None]:
    """
    A much faster reverse read of file by using Python's mmap to generate a
    memory-mapped file. It is slower for very small files than
    reverse_readline, but at least 2x faster for large files (the primary use
    of such a method).

    Args:
        filename (str):
            Name of file to read.

    Yields:
        Lines from the file in reverse order.
    """
    try:
        with zopen(filename, 'rb') as f:
            if isinstance(f, (gzip.GzipFile, bz2.BZ2File)):
                for line in reversed(f.readlines()):
                    yield line.decode('utf-8').rstrip()
            else:
                fm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                n = len(fm)
                while n > 0:
                    i = fm.rfind(b'\n', 0, n)
                    yield fm[i + 1:n].decode('utf-8').strip('\n')
                    n = i
    except ValueError:
        return