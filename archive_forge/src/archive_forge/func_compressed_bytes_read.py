from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
def compressed_bytes_read(self):
    """
        :return: number of compressed bytes read. This includes the bytes it
            took to decompress the header ( if there was one )"""
    if self._br == self._s and (not self._zip.unused_data):
        self._br = 0
        if hasattr(self._zip, 'status'):
            while self._zip.status == zlib.Z_OK:
                self.read(mmap.PAGESIZE)
        else:
            while not self._zip.unused_data and self._cbr != len(self._m):
                self.read(mmap.PAGESIZE)
        self._br = self._s
    return self._cbr