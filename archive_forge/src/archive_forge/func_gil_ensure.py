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
def gil_ensure(self):
    """
        Ensure the GIL is acquired.
        The returned value must be consumed by gil_release().
        """
    gilptrty = ir.PointerType(self.gil_state)
    fnty = ir.FunctionType(ir.VoidType(), [gilptrty])
    fn = self._get_function(fnty, 'numba_gil_ensure')
    gilptr = cgutils.alloca_once(self.builder, self.gil_state)
    self.builder.call(fn, [gilptr])
    return gilptr