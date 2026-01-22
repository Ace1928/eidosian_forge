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
def iter_sha1(iter):
    """Return the hexdigest of the SHA1 over a set of names.

    Args:
      iter: Iterator over string objects
    Returns: 40-byte hex sha1 digest
    """
    sha = sha1()
    for name in iter:
        sha.update(name)
    return sha.hexdigest().encode('ascii')