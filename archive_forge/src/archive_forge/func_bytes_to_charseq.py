import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@lower_cast(types.Bytes, types.CharSeq)
def bytes_to_charseq(context, builder, fromty, toty, val):
    barr = cgutils.create_struct_proxy(fromty)(context, builder, value=val)
    src = builder.bitcast(barr.data, ir.IntType(8).as_pointer())
    src_length = barr.nitems
    lty = context.get_value_type(toty)
    dstint_t = ir.IntType(8)
    dst_ptr = cgutils.alloca_once(builder, lty)
    dst = builder.bitcast(dst_ptr, dstint_t.as_pointer())
    dst_length = ir.Constant(src_length.type, toty.count)
    is_shorter_value = builder.icmp_unsigned('<', src_length, dst_length)
    count = builder.select(is_shorter_value, src_length, dst_length)
    with builder.if_then(is_shorter_value):
        cgutils.memset(builder, dst, ir.Constant(src_length.type, toty.count), 0)
    with cgutils.for_range(builder, count) as loop:
        in_ptr = builder.gep(src, [loop.index])
        in_val = builder.zext(builder.load(in_ptr), dstint_t)
        builder.store(in_val, builder.gep(dst, [loop.index]))
    return builder.load(dst_ptr)