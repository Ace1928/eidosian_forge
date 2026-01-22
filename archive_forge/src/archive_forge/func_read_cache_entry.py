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
def read_cache_entry(f, version: int) -> SerializedIndexEntry:
    """Read an entry from a cache file.

    Args:
      f: File-like object to read from
    """
    beginoffset = f.tell()
    ctime = read_cache_time(f)
    mtime = read_cache_time(f)
    dev, ino, mode, uid, gid, size, sha, flags = struct.unpack('>LLLLLL20sH', f.read(20 + 4 * 6 + 2))
    if flags & FLAG_EXTENDED:
        if version < 3:
            raise AssertionError('extended flag set in index with version < 3')
        extended_flags, = struct.unpack('>H', f.read(2))
    else:
        extended_flags = 0
    name = f.read(flags & FLAG_NAMEMASK)
    if version < 4:
        real_size = f.tell() - beginoffset + 8 & ~7
        f.read(beginoffset + real_size - f.tell())
    return SerializedIndexEntry(name, ctime, mtime, dev, ino, mode, uid, gid, size, sha_to_hex(sha), flags & ~FLAG_NAMEMASK, extended_flags)