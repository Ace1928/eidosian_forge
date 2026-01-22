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
def list_pack(self, items):
    n = len(items)
    seq = self.list_new(self.context.get_constant(types.intp, n))
    with self.if_object_ok(seq):
        for i in range(n):
            idx = self.context.get_constant(types.intp, i)
            self.incref(items[i])
            self.list_setitem(seq, idx, items[i])
    return seq