from numba.core import types, errors, cgutils
from numba.core.extending import intrinsic
@intrinsic
def exception_check(typingctx):
    """An intrinsic to check if an exception is raised
    """

    def codegen(context, builder, signature, args):
        nrt = context.nrt
        return nrt.eh_check(builder)
    restype = types.boolean
    return (restype(), codegen)