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
def index_entry_from_path(path: bytes, object_store: Optional[ObjectContainer]=None) -> Optional[IndexEntry]:
    """Create an index from a filesystem path.

    This returns an index value for files, symlinks
    and tree references. for directories and
    non-existent files it returns None

    Args:
      path: Path to create an index entry for
      object_store: Optional object store to
        save new blobs in
    Returns: An index entry; None for directories
    """
    assert isinstance(path, bytes)
    st = os.lstat(path)
    if stat.S_ISDIR(st.st_mode):
        return index_entry_from_directory(st, path)
    if stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode):
        blob = blob_from_path_and_stat(path, st)
        if object_store is not None:
            object_store.add_object(blob)
        return index_entry_from_stat(st, blob.id)
    return None