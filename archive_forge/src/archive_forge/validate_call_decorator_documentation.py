from __future__ import annotations as _annotations
import functools
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from ._internal import _validate_call
Usage docs: https://docs.pydantic.dev/2.6/concepts/validation_decorator/

    Returns a decorated wrapper around the function that validates the arguments and, optionally, the return value.

    Usage may be either as a plain decorator `@validate_call` or with arguments `@validate_call(...)`.

    Args:
        __func: The function to be decorated.
        config: The configuration dictionary.
        validate_return: Whether to validate the return value.

    Returns:
        The decorated function.
    