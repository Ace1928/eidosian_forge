import collections.abc
import copy
import dataclasses
import inspect
import sys
import types
import warnings
from typing import (
from typing_extensions import Annotated, Self, get_args, get_origin, get_type_hints
from . import _fields, _unsafe_cache
from ._typing import TypeForm
@_unsafe_cache.unsafe_cache(maxsize=1024)
def narrow_union_type(typ: TypeOrCallable, default_instance: Any) -> TypeOrCallable:
    """Narrow union types.

    This is a shim for failing more gracefully when we we're given a Union type that
    doesn't match the default value.

    In this case, we raise a warning, then add the type of the default value to the
    union. Loosely motivated by: https://github.com/brentyi/tyro/issues/20
    """
    if get_origin(typ) is not Union:
        return typ
    options = get_args(typ)
    options_unwrapped = [unwrap_origin_strip_extras(o) for o in options]
    try:
        if default_instance not in _fields.MISSING_SINGLETONS and (not any((isinstance(default_instance, o) for o in options_unwrapped))):
            warnings.warn(f'{type(default_instance)} does not match any type in Union: {options_unwrapped}')
            return Union.__getitem__(options + (type(default_instance),))
    except TypeError:
        pass
    return typ