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
def load_stack_global(self):
    name = self.stack.pop()
    module = self.stack.pop()
    if type(name) is not str or type(module) is not str:
        raise UnpicklingError('STACK_GLOBAL requires str')
    self.append(self.find_class(module, name))