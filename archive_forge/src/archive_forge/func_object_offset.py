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
def object_offset(self, sha: bytes) -> int:
    """Return the offset in to the corresponding packfile for the object.

        Given the name of an object it will return the offset that object
        lives at within the corresponding pack file. If the pack file doesn't
        have the object then None will be returned.
        """
    if len(sha) == 40:
        sha = hex_to_sha(sha)
    try:
        return self._object_offset(sha)
    except ValueError as exc:
        closed = getattr(self._contents, 'closed', None)
        if closed in (None, True):
            raise PackFileDisappeared(self) from exc
        raise