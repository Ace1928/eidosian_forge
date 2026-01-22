from __future__ import annotations
from .. import mesonlib, mparser
from .exceptions import InterpreterException, InvalidArguments
from ..coredata import UserOption
import collections.abc
import typing as T
def default_resolve_key(key: mparser.BaseNode) -> str:
    if not isinstance(key, mparser.IdNode):
        raise InterpreterException('Invalid kwargs format.')
    return key.value