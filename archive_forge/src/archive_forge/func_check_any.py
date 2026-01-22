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
def check_any(self, value: T.Any) -> bool:
    """Check a value should emit new/deprecated feature.

        :param value: A value to check
        :return: True if any of the items in value matches, False otherwise
        """
    if not isinstance(value, self.container):
        return False
    iter_ = iter(value.values()) if isinstance(value, dict) else iter(value)
    return any((isinstance(i, self.contains) for i in iter_))