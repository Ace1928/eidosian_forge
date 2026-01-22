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
def object_reset_private_data(self, obj):
    fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
    fn = self._get_function(fnty, name='numba_reset_pyobject_private_data')
    return self.builder.call(fn, (obj,))