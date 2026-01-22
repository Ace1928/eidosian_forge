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
def call_function_objargs(self, callee, objargs):
    fnty = ir.FunctionType(self.pyobj, [self.pyobj], var_arg=True)
    fn = self._get_function(fnty, name='PyObject_CallFunctionObjArgs')
    args = [callee] + list(objargs)
    args.append(self.context.get_constant_null(types.pyobject))
    return self.builder.call(fn, args)