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
def load_inst(self):
    module = self.readline()[:-1].decode('ascii')
    name = self.readline()[:-1].decode('ascii')
    klass = self.find_class(module, name)
    self._instantiate(klass, self.pop_mark())