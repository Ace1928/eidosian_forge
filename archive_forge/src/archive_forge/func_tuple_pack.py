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
def tuple_pack(self, items):
    fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t], var_arg=True)
    fn = self._get_function(fnty, name='PyTuple_Pack')
    n = self.context.get_constant(types.intp, len(items))
    args = [n]
    args.extend(items)
    return self.builder.call(fn, args)