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
def save_pers(self, pid):
    if self.bin:
        self.save(pid, save_persistent_id=False)
        self.write(BINPERSID)
    else:
        try:
            self.write(PERSID + str(pid).encode('ascii') + b'\n')
        except UnicodeEncodeError:
            raise PicklingError('persistent IDs in protocol 0 must be ASCII strings')