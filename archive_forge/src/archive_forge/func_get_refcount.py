from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import intrinsic
from numba.core.runtime.nrtdynmod import _meminfo_struct_type
@intrinsic
def get_refcount(typingctx, obj):
    """Get the current refcount of an object.

    FIXME: only handles the first object
    """

    def codegen(context, builder, signature, args):
        [obj] = args
        [ty] = signature.args
        meminfos = []
        if context.enable_nrt:
            tmp_mis = context.nrt.get_meminfos(builder, ty, obj)
            meminfos.extend(tmp_mis)
        refcounts = []
        if meminfos:
            for ty, mi in meminfos:
                miptr = builder.bitcast(mi, _meminfo_struct_type.as_pointer())
                refctptr = cgutils.gep_inbounds(builder, miptr, 0, 0)
                refct = builder.load(refctptr)
                refct_32bit = builder.trunc(refct, ir.IntType(32))
                refcounts.append(refct_32bit)
        return refcounts[0]
    sig = types.int32(obj)
    return (sig, codegen)