import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
def tuple_cmp_ordered(context, builder, op, sig, args):
    tu, tv = sig.args
    u, v = args
    res = cgutils.alloca_once_value(builder, cgutils.true_bit)
    bbend = builder.append_basic_block('cmp_end')
    for i, (ta, tb) in enumerate(zip(tu.types, tv.types)):
        a = builder.extract_value(u, i)
        b = builder.extract_value(v, i)
        not_equal = context.generic_compare(builder, operator.ne, (ta, tb), (a, b))
        with builder.if_then(not_equal):
            pred = context.generic_compare(builder, op, (ta, tb), (a, b))
            builder.store(pred, res)
            builder.branch(bbend)
    len_compare = op(len(tu.types), len(tv.types))
    pred = context.get_constant(types.boolean, len_compare)
    builder.store(pred, res)
    builder.branch(bbend)
    builder.position_at_end(bbend)
    return builder.load(res)