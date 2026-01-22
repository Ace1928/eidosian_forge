import os
from io import BytesIO
import zipfile
import tempfile
import shutil
import enum
import warnings
from ..core import urlopen, get_remote_file
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
def read_n_bytes(f, N):
    """read_n_bytes(file, n)

    Read n bytes from the given file, or less if the file has less
    bytes. Returns zero bytes if the file is closed.
    """
    bb = bytes()
    while len(bb) < N:
        extra_bytes = f.read(N - len(bb))
        if not extra_bytes:
            break
        bb += extra_bytes
    return bb