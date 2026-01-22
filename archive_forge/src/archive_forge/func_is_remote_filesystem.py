import importlib
import shutil
import threading
import warnings
from typing import List
import fsspec
import fsspec.asyn
from fsspec.implementations.local import LocalFileSystem
from ..utils.deprecation_utils import deprecated
from . import compression
def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    """
    Checks if `fs` is a remote filesystem.

    Args:
        fs (`fsspec.spec.AbstractFileSystem`):
            An abstract super-class for pythonic file-systems, e.g. `fsspec.filesystem('file')` or [`datasets.filesystems.S3FileSystem`].
    """
    return not isinstance(fs, LocalFileSystem)