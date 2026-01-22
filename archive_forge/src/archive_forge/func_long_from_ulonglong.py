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
def long_from_ulonglong(self, ival):
    return self._long_from_native_int(ival, 'PyLong_FromUnsignedLongLong', self.ulonglong, signed=False)