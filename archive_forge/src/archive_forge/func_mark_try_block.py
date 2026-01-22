from numba.core import types, errors, cgutils
from numba.core.extending import intrinsic
@intrinsic
def mark_try_block(typingctx):
    """An intrinsic to mark the start of a *try* block.
    """

    def codegen(context, builder, signature, args):
        nrt = context.nrt
        nrt.eh_try(builder)
        return context.get_dummy_value()
    restype = types.none
    return (restype(), codegen)