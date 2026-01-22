from __future__ import annotations
from collections.abc import Callable, Sequence
from functools import partial
from inspect import getmro, isclass
from typing import TYPE_CHECKING, Generic, Type, TypeVar, cast, overload
def subgroup(self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...] | Callable[[_ExceptionT_co], bool]) -> ExceptionGroup[_ExceptionT] | None:
    return super().subgroup(__condition)