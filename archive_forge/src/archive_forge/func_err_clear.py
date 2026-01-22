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
def err_clear(self):
    fnty = ir.FunctionType(ir.VoidType(), ())
    fn = self._get_function(fnty, name='PyErr_Clear')
    return self.builder.call(fn, ())