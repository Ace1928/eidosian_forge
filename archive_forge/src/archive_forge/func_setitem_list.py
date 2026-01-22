import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@lower_builtin(operator.setitem, types.List, types.SliceType, types.Any)
def setitem_list(context, builder, sig, args):
    dest = ListInstance(context, builder, sig.args[0], args[0])
    src = ListInstance(context, builder, sig.args[2], args[2])
    slice = context.make_helper(builder, sig.args[1], args[1])
    slicing.guard_invalid_slice(context, builder, sig.args[1], slice)
    dest.fix_slice(slice)
    src_size = src.size
    avail_size = slicing.get_slice_length(builder, slice)
    size_delta = builder.sub(src.size, avail_size)
    zero = ir.Constant(size_delta.type, 0)
    one = ir.Constant(size_delta.type, 1)
    with builder.if_else(builder.icmp_signed('==', slice.step, one)) as (then, otherwise):
        with then:
            real_stop = builder.add(slice.start, avail_size)
            tail_size = builder.sub(dest.size, real_stop)
            with builder.if_then(builder.icmp_signed('>', size_delta, zero)):
                dest.resize(builder.add(dest.size, size_delta))
                dest.move(builder.add(real_stop, size_delta), real_stop, tail_size)
            with builder.if_then(builder.icmp_signed('<', size_delta, zero)):
                dest.move(builder.add(real_stop, size_delta), real_stop, tail_size)
                dest.resize(builder.add(dest.size, size_delta))
            dest_offset = slice.start
            with cgutils.for_range(builder, src_size) as loop:
                value = src.getitem(loop.index)
                dest.setitem(builder.add(loop.index, dest_offset), value, incref=True)
        with otherwise:
            with builder.if_then(builder.icmp_signed('!=', size_delta, zero)):
                msg = 'cannot resize extended list slice with step != 1'
                context.call_conv.return_user_exc(builder, ValueError, (msg,))
            with cgutils.for_range_slice_generic(builder, slice.start, slice.stop, slice.step) as (pos_range, neg_range):
                with pos_range as (index, count):
                    value = src.getitem(count)
                    dest.setitem(index, value, incref=True)
                with neg_range as (index, count):
                    value = src.getitem(count)
                    dest.setitem(index, value, incref=True)
    return context.get_dummy_value()