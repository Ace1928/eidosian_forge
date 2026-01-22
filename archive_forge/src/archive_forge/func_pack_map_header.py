from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def pack_map_header(self, n):
    if n >= 2 ** 32:
        raise ValueError
    self._pack_map_header(n)
    if self._autoreset:
        ret = self._buffer.getvalue()
        self._buffer = StringIO()
        return ret