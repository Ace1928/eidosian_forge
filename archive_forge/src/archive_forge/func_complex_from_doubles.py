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
def complex_from_doubles(self, realval, imagval):
    fnty = ir.FunctionType(self.pyobj, [ir.DoubleType(), ir.DoubleType()])
    fn = self._get_function(fnty, name='PyComplex_FromDoubles')
    return self.builder.call(fn, [realval, imagval])