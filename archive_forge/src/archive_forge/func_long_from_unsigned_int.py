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
def long_from_unsigned_int(self, ival):
    """
        Same as long_from_signed_int, but for unsigned values.
        """
    bits = ival.type.width
    if bits <= self.ulong.width:
        return self.long_from_ulong(self.builder.zext(ival, self.ulong))
    elif bits <= self.ulonglong.width:
        return self.long_from_ulonglong(self.builder.zext(ival, self.ulonglong))
    else:
        raise OverflowError('integer too big (%d bits)' % bits)