from numba.core import types, errors, cgutils
from numba.core.extending import intrinsic
@intrinsic
def end_try_block(typingctx):
    """An intrinsic to mark the end of a *try* block.
    """

    def codegen(context, builder, signature, args):
        nrt = context.nrt
        nrt.eh_end_try(builder)
        return context.get_dummy_value()
    restype = types.none
    return (restype(), codegen)