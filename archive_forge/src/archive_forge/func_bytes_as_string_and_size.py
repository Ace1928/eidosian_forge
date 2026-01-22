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
def bytes_as_string_and_size(self, obj, p_buffer, p_length):
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.cstring.as_pointer(), self.py_ssize_t.as_pointer()])
    fname = 'PyBytes_AsStringAndSize'
    fn = self._get_function(fnty, name=fname)
    result = self.builder.call(fn, [obj, p_buffer, p_length])
    ok = self.builder.icmp_signed('!=', Constant(result.type, -1), result)
    return ok