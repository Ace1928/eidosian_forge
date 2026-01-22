from __future__ import absolute_import, division, print_function
import bz2
import hashlib
import logging
import os
import re
import struct
import sys
import types
import zlib
from io import BytesIO
def update_checksum(self):
    """Reset the blob's checksum if present. Call this after modifying
        the data.
        """
    if self.use_checksum and self._modified:
        self.seek(0)
        compressed = self._f.read(self.used_size)
        self._f.seek(self.start_pos - self.alignment - 1 - 16)
        self._f.write(hashlib.md5(compressed).digest())