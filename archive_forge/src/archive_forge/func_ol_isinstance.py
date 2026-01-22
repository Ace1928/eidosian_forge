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
@overload(isinstance)
def ol_isinstance(var, typs):

    def true_impl(var, typs):
        return True

    def false_impl(var, typs):
        return False
    var_ty = as_numba_type(var)
    if isinstance(var_ty, types.Optional):
        msg = f'isinstance cannot handle optional types. Found: "{var_ty}"'
        raise NumbaTypeError(msg)
    supported_var_ty = (types.Number, types.Bytes, types.RangeType, types.DictType, types.LiteralStrKeyDict, types.List, types.ListType, types.Tuple, types.UniTuple, types.Set, types.Function, types.ClassType, types.UnicodeType, types.ClassInstanceType, types.NoneType, types.Array, types.Boolean, types.Float, types.UnicodeCharSeq, types.Complex)
    if not isinstance(var_ty, supported_var_ty):
        msg = f'isinstance() does not support variables of type "{var_ty}".'
        raise NumbaTypeError(msg)
    t_typs = typs
    if isinstance(t_typs, types.UniTuple):
        t_typs = t_typs.key[0]
    if not isinstance(t_typs, types.Tuple):
        t_typs = (t_typs,)
    for typ in t_typs:
        if isinstance(typ, types.Function):
            key = typ.key[0]
        elif isinstance(typ, types.ClassType):
            key = typ
        else:
            key = typ.key
        types_not_registered = {bytes: types.Bytes, range: types.RangeType, dict: (types.DictType, types.LiteralStrKeyDict), list: types.List, tuple: types.BaseTuple, set: types.Set}
        if key in types_not_registered:
            if isinstance(var_ty, types_not_registered[key]):
                return true_impl
            continue
        if isinstance(typ, types.TypeRef):
            if key not in (types.ListType, types.DictType):
                msg = 'Numba type classes (except numba.typed.* container types) are not supported.'
                raise NumbaTypeError(msg)
            return true_impl if type(var_ty) is key else false_impl
        else:
            numba_typ = as_numba_type(key)
            if var_ty == numba_typ:
                return true_impl
            elif isinstance(numba_typ, types.ClassType) and isinstance(var_ty, types.ClassInstanceType) and (var_ty.key == numba_typ.instance_type.key):
                return true_impl
            elif isinstance(numba_typ, types.Container) and numba_typ.key[0] == types.undefined:
                if isinstance(var_ty, numba_typ.__class__) or (isinstance(var_ty, types.BaseTuple) and isinstance(numba_typ, types.BaseTuple)):
                    return true_impl
    return false_impl