from collections import namedtuple
import math
from functools import reduce
import numpy as np
import operator
import warnings
from llvmlite import ir
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, cgutils
from numba.core.extending import overload, intrinsic
from numba.core.typeconv import Conversion
from numba.core.errors import (TypingError, LoweringError,
from numba.misc.special import literal_unroll
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.typing.builtins import IndexValue, IndexValueType
from numba.extending import overload, register_jitable
@lower_builtin(types.TypeRef, types.VarArg(types.Any))
def redirect_type_ctor(context, builder, sig, args):
    """Redirect constructor implementation to `numba_typeref_ctor(cls, *args)`,
    which should be overloaded by the type's implementation.

    For example:

        d = Dict()

    `d` will be typed as `TypeRef[DictType]()`.  Thus, it will call into this
    implementation.  We need to redirect the lowering to a function
    named ``numba_typeref_ctor``.
    """
    cls = sig.return_type

    def call_ctor(cls, *args):
        return numba_typeref_ctor(cls, *args)
    ctor_args = types.Tuple.from_types(sig.args)
    sig = typing.signature(cls, types.TypeRef(cls), ctor_args)
    if len(ctor_args) > 0:
        args = (context.get_dummy_value(), context.make_tuple(builder, ctor_args, args))
    else:
        args = (context.get_dummy_value(), context.make_tuple(builder, ctor_args, ()))
    return context.compile_internal(builder, call_ctor, sig, args)