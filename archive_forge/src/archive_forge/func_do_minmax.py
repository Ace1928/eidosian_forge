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
def do_minmax(context, builder, argtys, args, cmpop):
    assert len(argtys) == len(args), (argtys, args)
    assert len(args) > 0

    def binary_minmax(accumulator, value):
        accty, acc = accumulator
        vty, v = value
        ty = context.typing_context.unify_types(accty, vty)
        assert ty is not None
        acc = context.cast(builder, acc, accty, ty)
        v = context.cast(builder, v, vty, ty)
        cmpsig = typing.signature(types.boolean, ty, ty)
        ge = context.get_function(cmpop, cmpsig)
        pred = ge(builder, (v, acc))
        res = builder.select(pred, v, acc)
        return (ty, res)
    typvals = zip(argtys, args)
    resty, resval = reduce(binary_minmax, typvals)
    return resval