from __future__ import annotations
from typing import Any, TypeVar, Iterable, cast
from collections import abc as _c_abc
from typing_extensions import Required, Annotated, get_args, get_origin
from .._types import InheritsGeneric
from .._compat import is_union as _is_union
def extract_type_arg(typ: type, index: int) -> type:
    args = get_args(typ)
    try:
        return cast(type, args[index])
    except IndexError as err:
        raise RuntimeError(f'Expected type {typ} to have a type argument at index {index} but it did not') from err