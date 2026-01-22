import binascii
from collections import defaultdict, deque
from contextlib import suppress
from io import BytesIO, UnsupportedOperation
import os
import struct
import sys
from itertools import chain
from typing import (
import warnings
import zlib
from hashlib import sha1
from os import SEEK_CUR, SEEK_END
from struct import unpack_from
from .errors import ApplyDeltaError, ChecksumMismatch
from .file import GitFile
from .lru_cache import LRUSizeCache
from .objects import ObjectID, ShaFile, hex_to_sha, object_header, sha_to_hex
def take_msb_bytes(read: Callable[[int], bytes], crc32: Optional[int]=None) -> Tuple[List[int], Optional[int]]:
    """Read bytes marked with most significant bit.

    Args:
      read: Read function
    """
    ret: List[int] = []
    while len(ret) == 0 or ret[-1] & 128:
        b = read(1)
        if crc32 is not None:
            crc32 = binascii.crc32(b, crc32)
        ret.append(ord(b[:1]))
    return (ret, crc32)