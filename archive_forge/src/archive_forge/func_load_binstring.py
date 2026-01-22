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
def load_binstring(self):
    len, = unpack('<i', self.read(4))
    if len < 0:
        raise UnpicklingError('BINSTRING pickle has negative byte count')
    data = self.read(len)
    self.append(self._decode_string(data))