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
def number_power(self, lhs, rhs, inplace=False):
    fnty = ir.FunctionType(self.pyobj, [self.pyobj] * 3)
    fname = 'PyNumber_InPlacePower' if inplace else 'PyNumber_Power'
    fn = self._get_function(fnty, fname)
    return self.builder.call(fn, [lhs, rhs, self.borrow_none()])