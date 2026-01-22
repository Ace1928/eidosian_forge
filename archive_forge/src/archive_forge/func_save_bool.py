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
def save_bool(self, obj):
    if self.proto >= 2:
        self.write(NEWTRUE if obj else NEWFALSE)
    else:
        self.write(TRUE if obj else FALSE)