from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def save_bytes(self, obj):
    if self.proto < 3:
        if not obj:
            self.save_reduce(bytes, (), obj=obj)
        else:
            self.save_reduce(codecs.encode, (str(obj, 'latin1'), 'latin1'), obj=obj)
        return
    n = len(obj)
    if n <= 255:
        self.write(SHORT_BINBYTES + pack('<B', n) + obj)
    elif n > 4294967295 and self.proto >= 4:
        self._write_large_bytes(BINBYTES8 + pack('<Q', n), obj)
    elif n >= self.framer._FRAME_SIZE_TARGET:
        self._write_large_bytes(BINBYTES + pack('<I', n), obj)
    else:
        self.write(BINBYTES + pack('<I', n) + obj)
    self.memoize(obj)