from __future__ import annotations
import sys
import warnings
from typing import Any, Callable, NoReturn, TypeVar, Union, overload
from . import _suppression
from ._checkers import BINARY_MAGIC_METHODS, check_type_internal
from ._config import (
from ._exceptions import TypeCheckError, TypeCheckWarning
from ._memo import TypeCheckMemo
from ._utils import get_stacklevel, qualified_name
def check_yield_type(func_name: str, yieldval: T, annotation: Any, memo: TypeCheckMemo) -> T:
    if _suppression.type_checks_suppressed:
        return yieldval
    if annotation is NoReturn or annotation is Never:
        exc = TypeCheckError(f'{func_name}() was declared never to yield but it did')
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise exc
    try:
        check_type_internal(yieldval, annotation, memo)
    except TypeCheckError as exc:
        qualname = qualified_name(yieldval, add_class_prefix=True)
        exc.append_path_element(f'the yielded value ({qualname})')
        if memo.config.typecheck_fail_callback:
            memo.config.typecheck_fail_callback(exc, memo)
        else:
            raise
    return yieldval