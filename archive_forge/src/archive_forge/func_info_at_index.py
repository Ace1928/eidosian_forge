import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
def info_at_index(self, index):
    """As ``info``, but uses a PackIndexFile compatible index to refer to the object"""
    return self._object(None, False, index)