from __future__ import annotations as _annotations
import dataclasses
import sys
import warnings
from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from pydantic_core import PydanticUndefined
from pydantic.errors import PydanticUserError
from . import _typing_extra
from ._config import ConfigWrapper
from ._repr import Representation
from ._typing_extra import get_cls_type_hints_lenient, get_type_hints, is_classvar, is_finalvar
def get_type_hints_infer_globalns(obj: Any, localns: dict[str, Any] | None=None, include_extras: bool=False) -> dict[str, Any]:
    """Gets type hints for an object by inferring the global namespace.

    It uses the `typing.get_type_hints`, The only thing that we do here is fetching
    global namespace from `obj.__module__` if it is not `None`.

    Args:
        obj: The object to get its type hints.
        localns: The local namespaces.
        include_extras: Whether to recursively include annotation metadata.

    Returns:
        The object type hints.
    """
    module_name = getattr(obj, '__module__', None)
    globalns: dict[str, Any] | None = None
    if module_name:
        try:
            globalns = sys.modules[module_name].__dict__
        except KeyError:
            pass
    return get_type_hints(obj, globalns=globalns, localns=localns, include_extras=include_extras)