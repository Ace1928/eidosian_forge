from collections.abc import MutableMapping, Iterable, Mapping
from numba.core.types import DictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
from numba.typed import dictobject
from numba.core.typing import signature
@type_callable(DictType)
def typeddict_call(context):
    """
    Defines typing logic for ``Dict()`` and ``Dict(iterable)``.
    Produces Dict[undefined, undefined] or Dict[key, value]
    """

    def typer(arg=None):
        if arg is None:
            return types.DictType(types.undefined, types.undefined)
        elif isinstance(arg, types.DictType):
            return arg
        elif isinstance(arg, types.Tuple) and len(arg) == 0:
            msg = "non-precise type 'dict(())'"
            raise errors.TypingError(msg)
        elif isinstance(arg, types.IterableType):
            dtype = arg.iterator_type.yield_type
            if isinstance(dtype, types.UniTuple):
                key = value = dtype.key[0]
                return types.DictType(key, value)
            elif isinstance(dtype, types.Tuple):
                key, value = dtype.key
                return types.DictType(key, value)
    return typer