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
def check_protocol(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:
    if getattr(origin_type, '_is_runtime_protocol', False):
        if not isinstance(value, origin_type):
            raise TypeCheckError(f'is not compatible with the {origin_type.__qualname__} protocol')
    else:
        warnings.warn(f'Typeguard cannot check the {origin_type.__qualname__} protocol because it is a non-runtime protocol. If you would like to type check this protocol, please use @typing.runtime_checkable', stacklevel=get_stacklevel())