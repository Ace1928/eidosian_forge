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
def load_pack_index(path):
    """Load an index file by path.

    Args:
      path: Path to the index file
    Returns: A PackIndex loaded from the given path
    """
    with GitFile(path, 'rb') as f:
        return load_pack_index_file(path, f)