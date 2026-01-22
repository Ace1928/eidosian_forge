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
def check_io(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    if origin_type is TextIO or (origin_type is IO and args == (str,)):
        if not isinstance(value, TextIOBase):
            raise TypeCheckError('is not a text based I/O object')
    elif origin_type is BinaryIO or (origin_type is IO and args == (bytes,)):
        if not isinstance(value, (RawIOBase, BufferedIOBase)):
            raise TypeCheckError('is not a binary I/O object')
    elif not isinstance(value, IOBase):
        raise TypeCheckError('is not an I/O object')