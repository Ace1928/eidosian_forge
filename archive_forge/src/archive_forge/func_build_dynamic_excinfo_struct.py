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
def build_dynamic_excinfo_struct(self, struct_gv, exc_args):
    """
        Serialize some data at runtime. Returns a pointer to a python tuple
        (bytes_data, hash) where the first element is the serialized data as
        bytes and the second its hash.
        """
    fnty = ir.FunctionType(self.pyobj, (self.pyobj, self.pyobj))
    fn = self._get_function(fnty, name='numba_runtime_build_excinfo_struct')
    return self.builder.call(fn, (struct_gv, exc_args))