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
@classmethod
def from_lazy_objects(cls, data_fn, idx_fn):
    """Create a new pack object from callables to load pack data and
        index objects.
        """
    ret = cls('')
    ret._data_load = data_fn
    ret._idx_load = idx_fn
    return ret