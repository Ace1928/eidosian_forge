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
def from_native_value(self, typ, val, env_manager=None):
    """
        Box the native value of the given Numba type.  A Python object
        pointer is returned (NULL if an error occurred).
        This method steals any native (NRT) reference embedded in *val*.
        """
    from numba.core.boxing import box_unsupported
    impl = _boxers.lookup(typ.__class__, box_unsupported)
    c = _BoxContext(self.context, self.builder, self, env_manager)
    return impl(typ, val, c)