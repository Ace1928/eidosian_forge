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
def write_pack_header(write, num_objects):
    """Write a pack header for the given number of objects."""
    if hasattr(write, 'write'):
        write = write.write
        warnings.warn('write_pack_header() now takes a write rather than file argument', DeprecationWarning, stacklevel=2)
    for chunk in pack_header_chunks(num_objects):
        write(chunk)