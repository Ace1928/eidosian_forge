from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def unpack_exception(self, builder, pyapi, status):
    excinfo_ptr = status.excinfoptr
    alloc_flag = builder.extract_value(builder.load(excinfo_ptr), ALLOC_FLAG_IDX)
    gt = builder.icmp_signed('>', alloc_flag, int32_t(0))
    with builder.if_else(gt) as (then, otherwise):
        with then:
            dyn_exc = self.unpack_dynamic_exception(builder, pyapi, status)
            bb_then = builder.block
        with otherwise:
            static_exc = pyapi.unserialize(excinfo_ptr)
            bb_else = builder.block
    phi = builder.phi(static_exc.type)
    phi.add_incoming(dyn_exc, bb_then)
    phi.add_incoming(static_exc, bb_else)
    return phi