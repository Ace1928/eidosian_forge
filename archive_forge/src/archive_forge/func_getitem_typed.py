import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin('typed_getitem', types.BaseTuple, types.Any)
def getitem_typed(context, builder, sig, args):
    tupty, _ = sig.args
    tup, idx = args
    errmsg_oob = ('tuple index out of range',)
    if len(tupty) == 0:
        with builder.if_then(cgutils.true_bit):
            context.call_conv.return_user_exc(builder, IndexError, errmsg_oob)
        res = context.get_constant_null(sig.return_type)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    else:
        bbelse = builder.append_basic_block('typed_switch.else')
        bbend = builder.append_basic_block('typed_switch.end')
        switch = builder.switch(idx, bbelse)
        with builder.goto_block(bbelse):
            context.call_conv.return_user_exc(builder, IndexError, errmsg_oob)
        lrtty = context.get_value_type(sig.return_type)
        voidptrty = context.get_value_type(types.voidptr)
        with builder.goto_block(bbend):
            phinode = builder.phi(voidptrty)
        for i in range(tupty.count):
            ki = context.get_constant(types.intp, i)
            bbi = builder.append_basic_block('typed_switch.%d' % i)
            switch.add_case(ki, bbi)
            kin = context.get_constant(types.intp, -tupty.count + i)
            switch.add_case(kin, bbi)
            with builder.goto_block(bbi):
                value = builder.extract_value(tup, i)
                DOCAST = context.typing_context.unify_types(sig.args[0][i], sig.return_type) == sig.return_type
                if DOCAST:
                    value_slot = builder.alloca(lrtty, name='TYPED_VALUE_SLOT%s' % i)
                    casted = context.cast(builder, value, sig.args[0][i], sig.return_type)
                    builder.store(casted, value_slot)
                else:
                    value_slot = builder.alloca(value.type, name='TYPED_VALUE_SLOT%s' % i)
                    builder.store(value, value_slot)
                phinode.add_incoming(builder.bitcast(value_slot, voidptrty), bbi)
                builder.branch(bbend)
        builder.position_at_end(bbend)
        res = builder.bitcast(phinode, lrtty.as_pointer())
        res = builder.load(res)
        return impl_ret_borrowed(context, builder, sig.return_type, res)