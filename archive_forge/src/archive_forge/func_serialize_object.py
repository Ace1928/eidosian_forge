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
def serialize_object(self, obj):
    """
        Serialize the given object in the bitcode, and return it
        as a pointer to a
        {i8* data, i32 length, i8* hashbuf, i8* fn_ptr, i32 alloc_flag},
        structure constant (suitable for passing to unserialize()).
        """
    try:
        gv = self.module.__serialized[obj]
    except KeyError:
        struct = self.serialize_uncached(obj)
        name = '.const.picklebuf.%s' % (id(obj) if config.DIFF_IR == 0 else 'DIFF_IR')
        gv = self.context.insert_unique_const(self.module, name, struct)
        self.module.__serialized[obj] = gv
    return gv