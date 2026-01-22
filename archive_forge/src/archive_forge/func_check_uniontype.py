from __future__ import annotations
import collections.abc
import inspect
import sys
import types
import typing
import warnings
from enum import Enum
from inspect import Parameter, isclass, isfunction
from io import BufferedIOBase, IOBase, RawIOBase, TextIOBase
from textwrap import indent
from typing import (
from unittest.mock import Mock
from ._config import ForwardRefPolicy
from ._exceptions import TypeCheckError, TypeHintWarning
from ._memo import TypeCheckMemo
from ._utils import evaluate_forwardref, get_stacklevel, get_type_name, qualified_name
def check_uniontype(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    errors: dict[str, TypeCheckError] = {}
    for type_ in args:
        try:
            check_type_internal(value, type_, memo)
            return
        except TypeCheckError as exc:
            errors[get_type_name(type_)] = exc
    formatted_errors = indent('\n'.join((f'{key}: {error}' for key, error in errors.items())), '  ')
    raise TypeCheckError(f'did not match any element in the union:\n{formatted_errors}')