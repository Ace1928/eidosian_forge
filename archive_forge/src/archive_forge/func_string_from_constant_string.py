from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def string_from_constant_string(self, string):
    cstr = self.context.insert_const_string(self.module, string)
    sz = self.context.get_constant(types.intp, len(string))
    return self.string_from_string_and_size(cstr, sz)