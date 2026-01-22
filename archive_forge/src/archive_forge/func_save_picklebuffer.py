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
def save_picklebuffer(self, obj):
    if self.proto < 5:
        raise PicklingError('PickleBuffer can only pickled with protocol >= 5')
    with obj.raw() as m:
        if not m.contiguous:
            raise PicklingError('PickleBuffer can not be pickled when pointing to a non-contiguous buffer')
        in_band = True
        if self._buffer_callback is not None:
            in_band = bool(self._buffer_callback(obj))
        if in_band:
            if m.readonly:
                self.save_bytes(m.tobytes())
            else:
                self.save_bytearray(m.tobytes())
        else:
            self.write(NEXT_BUFFER)
            if m.readonly:
                self.write(READONLY_BUFFER)