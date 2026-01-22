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
def object_istrue(self, obj):
    fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
    fn = self._get_function(fnty, name='PyObject_IsTrue')
    return self.builder.call(fn, [obj])