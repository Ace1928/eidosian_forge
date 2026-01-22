from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32
@register_jitable
def next_float(bitgen):
    return float32(float32(next_uint32(bitgen) >> 8) * float32(1.0) / float32(16777216.0))