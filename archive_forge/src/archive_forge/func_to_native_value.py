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
def to_native_value(self, typ, obj):
    """
        Unbox the Python object as the given Numba type.
        A NativeValue instance is returned.
        """
    from numba.core.boxing import unbox_unsupported
    impl = _unboxers.lookup(typ.__class__, unbox_unsupported)
    c = _UnboxContext(self.context, self.builder, self)
    return impl(typ, obj, c)