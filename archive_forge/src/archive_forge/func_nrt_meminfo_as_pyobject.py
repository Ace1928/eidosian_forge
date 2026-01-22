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
def nrt_meminfo_as_pyobject(self, miptr):
    mod = self.builder.module
    fnty = ir.FunctionType(self.pyobj, [cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_meminfo_as_pyobject')
    fn.return_value.add_attribute('noalias')
    return self.builder.call(fn, [miptr])