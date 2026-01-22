import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
def fixed_enter(*args, **kwargs):
    return WrappedFile(_enter(*args, **kwargs))