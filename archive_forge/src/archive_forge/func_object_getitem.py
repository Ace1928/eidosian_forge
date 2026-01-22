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
def object_getitem(self, obj, key):
    """
        Return obj[key]
        """
    fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj])
    fn = self._get_function(fnty, name='PyObject_GetItem')
    return self.builder.call(fn, (obj, key))