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
def long_as_voidptr(self, numobj):
    """
        Convert the given Python integer to a void*.  This is recommended
        over number_as_ssize_t as it isn't affected by signedness.
        """
    fnty = ir.FunctionType(self.voidptr, [self.pyobj])
    fn = self._get_function(fnty, name='PyLong_AsVoidPtr')
    return self.builder.call(fn, [numobj])