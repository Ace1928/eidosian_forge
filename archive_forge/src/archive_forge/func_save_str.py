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
def save_str(self, obj):
    if self.bin:
        encoded = obj.encode('utf-8', 'surrogatepass')
        n = len(encoded)
        if n <= 255 and self.proto >= 4:
            self.write(SHORT_BINUNICODE + pack('<B', n) + encoded)
        elif n > 4294967295 and self.proto >= 4:
            self._write_large_bytes(BINUNICODE8 + pack('<Q', n), encoded)
        elif n >= self.framer._FRAME_SIZE_TARGET:
            self._write_large_bytes(BINUNICODE + pack('<I', n), encoded)
        else:
            self.write(BINUNICODE + pack('<I', n) + encoded)
    else:
        tmp = obj.replace('\\', '\\u005c')
        tmp = tmp.replace('\x00', '\\u0000')
        tmp = tmp.replace('\n', '\\u000a')
        tmp = tmp.replace('\r', '\\u000d')
        tmp = tmp.replace('\x1a', '\\u001a')
        self.write(UNICODE + tmp.encode('raw-unicode-escape') + b'\n')
    self.memoize(obj)