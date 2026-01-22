from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
@classmethod
def single_use(cls, feature_name: str, version: str, subproject: 'SubProject', extra_message: str='', location: T.Optional['mparser.BaseNode']=None) -> None:
    """Oneline version that instantiates and calls use()."""
    cls(feature_name, version, extra_message).use(subproject, location)