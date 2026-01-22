from collections.abc import MutableMapping, Iterable, Mapping
from numba.core.types import DictType
from numba.core.imputils import numba_typeref_ctor
from numba import njit, typeof
from numba.core import types, errors, config, cgutils
from numba.core.extending import (
from numba.typed import dictobject
from numba.core.typing import signature
@overload_classmethod(types.DictType, 'empty')
def typeddict_empty(cls, key_type, value_type, n_keys=0):
    if cls.instance_type is not DictType:
        return

    def impl(cls, key_type, value_type, n_keys=0):
        return dictobject.new_dict(key_type, value_type, n_keys=n_keys)
    return impl