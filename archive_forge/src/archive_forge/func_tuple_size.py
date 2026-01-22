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
def tuple_size(self, tup):
    fnty = ir.FunctionType(self.py_ssize_t, [self.pyobj])
    fn = self._get_function(fnty, name='PyTuple_Size')
    return self.builder.call(fn, [tup])