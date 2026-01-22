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
def extract_record_data(self, obj, pbuf):
    fnty = ir.FunctionType(self.voidptr, [self.pyobj, ir.PointerType(self.py_buffer_t)])
    fn = self._get_function(fnty, name='numba_extract_record_data')
    return self.builder.call(fn, [obj, pbuf])