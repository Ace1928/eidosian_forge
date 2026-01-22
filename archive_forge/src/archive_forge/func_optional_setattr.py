import operator
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_cast, lower_builtin,
@lower_setattr_generic(types.Optional)
def optional_setattr(context, builder, sig, args, attr):
    """
    Optional.__setattr__ => redirect to the wrapped type.
    """
    basety, valty = sig.args
    target, val = args
    target_type = basety.type
    target = context.cast(builder, target, basety, target_type)
    newsig = typing.signature(sig.return_type, target_type, valty)
    imp = context.get_setattr(attr, newsig)
    return imp(builder, (target, val))