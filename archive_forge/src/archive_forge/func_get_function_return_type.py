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
def get_function_return_type(func: Any, explicit_return_type: Any, types_namespace: dict[str, Any] | None=None) -> Any:
    """Get the function return type.

    It gets the return type from the type annotation if `explicit_return_type` is `None`.
    Otherwise, it returns `explicit_return_type`.

    Args:
        func: The function to get its return type.
        explicit_return_type: The explicit return type.
        types_namespace: The types namespace, defaults to `None`.

    Returns:
        The function return type.
    """
    if explicit_return_type is PydanticUndefined:
        hints = get_function_type_hints(unwrap_wrapped_function(func), include_keys={'return'}, types_namespace=types_namespace)
        return hints.get('return', PydanticUndefined)
    else:
        return explicit_return_type