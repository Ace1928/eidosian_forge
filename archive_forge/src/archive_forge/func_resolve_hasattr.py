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
@intrinsic
def resolve_hasattr(tyctx, obj, name):
    if not isinstance(name, types.StringLiteral):
        raise RequireLiteralValue("argument 'name' must be a literal string")
    lname = name.literal_value
    fn = tyctx.resolve_getattr(obj, lname)
    if fn is None:
        retty = types.literal(False)
    else:
        retty = types.literal(True)
    sig = retty(obj, name)

    def impl(cgctx, builder, sig, ll_args):
        return cgutils.false_bit if fn is None else cgutils.true_bit
    return (sig, impl)