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
def string_as_string(self, strobj):
    fnty = ir.FunctionType(self.cstring, [self.pyobj])
    fname = 'PyUnicode_AsUTF8'
    fn = self._get_function(fnty, name=fname)
    return self.builder.call(fn, [strobj])