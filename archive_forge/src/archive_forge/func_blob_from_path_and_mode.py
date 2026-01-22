import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def blob_from_path_and_mode(fs_path: bytes, mode: int, tree_encoding='utf-8'):
    """Create a blob from a path and a stat object.

    Args:
      fs_path: Full file system path to file
      mode: File mode
    Returns: A `Blob` object
    """
    assert isinstance(fs_path, bytes)
    blob = Blob()
    if stat.S_ISLNK(mode):
        if sys.platform == 'win32':
            blob.data = os.readlink(os.fsdecode(fs_path)).encode(tree_encoding)
        else:
            blob.data = os.readlink(fs_path)
    else:
        with open(fs_path, 'rb') as f:
            blob.data = f.read()
    return blob