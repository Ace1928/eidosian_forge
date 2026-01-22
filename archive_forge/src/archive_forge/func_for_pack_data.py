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
def for_pack_data(cls, pack_data: PackData, resolve_ext_ref=None):
    walker = cls(None, resolve_ext_ref=resolve_ext_ref)
    walker.set_pack_data(pack_data)
    for unpacked in pack_data.iter_unpacked(include_comp=False):
        walker.record(unpacked)
    return walker