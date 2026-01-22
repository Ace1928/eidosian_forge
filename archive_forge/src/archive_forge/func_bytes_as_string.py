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
def bytes_as_string(self, obj):
    fnty = ir.FunctionType(self.cstring, [self.pyobj])
    fname = 'PyBytes_AsString'
    fn = self._get_function(fnty, name=fname)
    return self.builder.call(fn, [obj])