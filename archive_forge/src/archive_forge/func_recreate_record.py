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
def recreate_record(self, pdata, size, dtype, env_manager):
    fnty = ir.FunctionType(self.pyobj, [ir.PointerType(ir.IntType(8)), ir.IntType(32), self.pyobj])
    fn = self._get_function(fnty, name='numba_recreate_record')
    dtypeaddr = env_manager.read_const(env_manager.add_const(dtype))
    return self.builder.call(fn, [pdata, size, dtypeaddr])