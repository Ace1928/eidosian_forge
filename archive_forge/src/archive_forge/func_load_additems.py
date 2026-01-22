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
def load_additems(self):
    items = self.pop_mark()
    set_obj = self.stack[-1]
    if isinstance(set_obj, set):
        set_obj.update(items)
    else:
        add = set_obj.add
        for item in items:
            add(item)