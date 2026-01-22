from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
def ensure_classmethod_based_on_signature(function: AnyDecoratorCallable) -> Any:
    """Apply the `@classmethod` decorator on the function.

    Args:
        function: The function to apply the decorator on.

    Return:
        The `@classmethod` decorator applied function.
    """
    if not isinstance(unwrap_wrapped_function(function, unwrap_class_static_method=False), classmethod) and _is_classmethod_from_sig(function):
        return classmethod(function)
    return function