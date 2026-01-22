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
def load_int(self):
    data = self.readline()
    if data == FALSE[1:]:
        val = False
    elif data == TRUE[1:]:
        val = True
    else:
        val = int(data, 0)
    self.append(val)