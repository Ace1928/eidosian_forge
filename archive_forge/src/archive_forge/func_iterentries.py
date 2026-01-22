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
def iterentries(self, progress=None, resolve_ext_ref: Optional[ResolveExtRefFn]=None):
    """Yield entries summarizing the contents of this pack.

        Args:
          progress: Progress function, called with current and total
            object count.
        Returns: iterator of tuples with (sha, offset, crc32)
        """
    num_objects = self._num_objects
    indexer = PackIndexer.for_pack_data(self, resolve_ext_ref=resolve_ext_ref)
    for i, result in enumerate(indexer):
        if progress is not None:
            progress(i, num_objects)
        yield result