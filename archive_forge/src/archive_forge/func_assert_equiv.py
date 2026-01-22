import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
@intrinsic
def assert_equiv(typingctx, *val):
    """
    A function that asserts the inputs are of equivalent size,
    and throws runtime error when they are not. The input is
    a vararg that contains an error message, followed by a set
    of objects of either array, tuple or integer.
    """
    if len(val) > 1:
        val = (types.StarArgTuple(val),)
    assert len(val[0]) > 1
    assert all((isinstance(a, (types.ArrayCompatible, types.BaseTuple, types.SliceType, types.Integer)) for a in val[0][1:]))
    if not isinstance(val[0][0], types.StringLiteral):
        raise errors.TypingError('first argument must be a StringLiteral')

    def codegen(context, builder, sig, args):
        assert len(args) == 1
        tup = cgutils.unpack_tuple(builder, args[0])
        tup_type = sig.args[0]
        msg = sig.args[0][0].literal_value

        def unpack_shapes(a, aty):
            if isinstance(aty, types.ArrayCompatible):
                ary = context.make_array(aty)(context, builder, a)
                return cgutils.unpack_tuple(builder, ary.shape)
            elif isinstance(aty, types.BaseTuple):
                return cgutils.unpack_tuple(builder, a)
            else:
                return [a]

        def pairwise(a, aty, b, bty):
            ashapes = unpack_shapes(a, aty)
            bshapes = unpack_shapes(b, bty)
            assert len(ashapes) == len(bshapes)
            for m, n in zip(ashapes, bshapes):
                m_eq_n = builder.icmp_unsigned('==', m, n)
                with builder.if_else(m_eq_n) as (then, orelse):
                    with then:
                        pass
                    with orelse:
                        context.call_conv.return_user_exc(builder, AssertionError, (msg,))
        for i in range(1, len(tup_type) - 1):
            pairwise(tup[i], tup_type[i], tup[i + 1], tup_type[i + 1])
        r = context.get_constant_generic(builder, types.NoneType, None)
        return r
    return (signature(types.none, *val), codegen)