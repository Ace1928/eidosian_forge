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
def err_format(self, exctype, msg, *format_args):
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj, self.cstring], var_arg=True)
    fn = self._get_function(fnty, name='PyErr_Format')
    if isinstance(exctype, str):
        exctype = self.get_c_object(exctype)
    if isinstance(msg, str):
        msg = self.context.insert_const_string(self.module, msg)
    return self.builder.call(fn, (exctype, msg) + tuple(format_args))