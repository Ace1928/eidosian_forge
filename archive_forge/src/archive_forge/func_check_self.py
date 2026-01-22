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
def check_self(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    if memo.self_type is None:
        raise TypeCheckError('cannot be checked against Self outside of a method call')
    if isclass(value):
        if not issubclass(value, memo.self_type):
            raise TypeCheckError(f'is not an instance of the self type ({qualified_name(memo.self_type)})')
    elif not isinstance(value, memo.self_type):
        raise TypeCheckError(f'is not an instance of the self type ({qualified_name(memo.self_type)})')