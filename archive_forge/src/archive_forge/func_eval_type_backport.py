from __future__ import annotations as _annotations
import dataclasses
import sys
import types
import typing
from collections.abc import Callable
from functools import partial
from types import GetSetDescriptorType
from typing import TYPE_CHECKING, Any, Final
from typing_extensions import Annotated, Literal, TypeAliasType, TypeGuard, get_args, get_origin
def eval_type_backport(value: Any, globalns: dict[str, Any] | None=None, localns: dict[str, Any] | None=None) -> Any:
    """Like `typing._eval_type`, but falls back to the `eval_type_backport` package if it's
    installed to let older Python versions use newer typing features.
    Specifically, this transforms `X | Y` into `typing.Union[X, Y]`
    and `list[X]` into `typing.List[X]` etc. (for all the types made generic in PEP 585)
    if the original syntax is not supported in the current Python version.
    """
    try:
        return typing._eval_type(value, globalns, localns)
    except TypeError as e:
        if not (isinstance(value, typing.ForwardRef) and is_backport_fixable_error(e)):
            raise
        try:
            from eval_type_backport import eval_type_backport
        except ImportError:
            raise TypeError(f'You have a type annotation {value.__forward_arg__!r} which makes use of newer typing features than are supported in your version of Python. To handle this error, you should either remove the use of new syntax or install the `eval_type_backport` package.') from e
        return eval_type_backport(value, globalns, localns, try_default=False)