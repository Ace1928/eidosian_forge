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
def inspect_field_serializer(serializer: Callable[..., Any], mode: Literal['plain', 'wrap'], computed_field: bool=False) -> tuple[bool, bool]:
    """Look at a field serializer function and determine if it is a field serializer,
    and whether it takes an info argument.

    An error is raised if the function has an invalid signature.

    Args:
        serializer: The serializer function to inspect.
        mode: The serializer mode, either 'plain' or 'wrap'.
        computed_field: When serializer is applied on computed_field. It doesn't require
            info signature.

    Returns:
        Tuple of (is_field_serializer, info_arg).
    """
    sig = signature(serializer)
    first = next(iter(sig.parameters.values()), None)
    is_field_serializer = first is not None and first.name == 'self'
    n_positional = count_positional_params(sig)
    if is_field_serializer:
        info_arg = _serializer_info_arg(mode, n_positional - 1)
    else:
        info_arg = _serializer_info_arg(mode, n_positional)
    if info_arg is None:
        raise PydanticUserError(f'Unrecognized field_serializer function signature for {serializer} with `mode={mode}`:{sig}', code='field-serializer-signature')
    if info_arg and computed_field:
        raise PydanticUserError('field_serializer on computed_field does not use info signature', code='field-serializer-signature')
    else:
        return (is_field_serializer, info_arg)