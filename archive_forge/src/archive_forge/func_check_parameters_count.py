from __future__ import annotations
import sys
import types
import typing
from collections import ChainMap
from contextlib import contextmanager
from contextvars import ContextVar
from types import prepare_class
from typing import TYPE_CHECKING, Any, Iterator, List, Mapping, MutableMapping, Tuple, TypeVar
from weakref import WeakValueDictionary
import typing_extensions
from ._core_utils import get_type_ref
from ._forward_ref import PydanticRecursiveRef
from ._typing_extra import TypeVarType, typing_base
from ._utils import all_identical, is_model_class
def check_parameters_count(cls: type[BaseModel], parameters: tuple[Any, ...]) -> None:
    """Check the generic model parameters count is equal.

    Args:
        cls: The generic model.
        parameters: A tuple of passed parameters to the generic model.

    Raises:
        TypeError: If the passed parameters count is not equal to generic model parameters count.
    """
    actual = len(parameters)
    expected = len(cls.__pydantic_generic_metadata__['parameters'])
    if actual != expected:
        description = 'many' if actual > expected else 'few'
        raise TypeError(f'Too {description} parameters for {cls}; actual {actual}, expected {expected}')