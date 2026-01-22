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
def inspect_validator(validator: Callable[..., Any], mode: FieldValidatorModes) -> bool:
    """Look at a field or model validator function and determine whether it takes an info argument.

    An error is raised if the function has an invalid signature.

    Args:
        validator: The validator function to inspect.
        mode: The proposed validator mode.

    Returns:
        Whether the validator takes an info argument.
    """
    try:
        sig = signature(validator)
    except ValueError:
        return False
    n_positional = count_positional_params(sig)
    if mode == 'wrap':
        if n_positional == 3:
            return True
        elif n_positional == 2:
            return False
    else:
        assert mode in {'before', 'after', 'plain'}, f"invalid mode: {mode!r}, expected 'before', 'after' or 'plain"
        if n_positional == 2:
            return True
        elif n_positional == 1:
            return False
    raise PydanticUserError(f'Unrecognized field_validator function signature for {validator} with `mode={mode}`:{sig}', code='validator-signature')