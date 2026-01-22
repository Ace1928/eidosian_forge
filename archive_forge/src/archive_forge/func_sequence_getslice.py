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
def sequence_getslice(self, obj, start, stop):
    fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.py_ssize_t, self.py_ssize_t])
    fn = self._get_function(fnty, name='PySequence_GetSlice')
    return self.builder.call(fn, (obj, start, stop))