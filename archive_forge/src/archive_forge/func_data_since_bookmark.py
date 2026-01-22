import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
def data_since_bookmark(self):
    assert self._bookmark is not None
    return self._buffer[self._bookmark:self._index]