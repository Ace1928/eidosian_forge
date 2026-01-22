from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar
import warnings
from markdown_it._compat import DATACLASS_KWARGS
from .utils import EnvType
def getRules(self, chainName: str='') -> list[RuleFuncTv]:
    """Return array of active functions (rules) for given chain name.
        It analyzes rules configuration, compiles caches if not exists and returns result.

        Default chain name is `''` (empty string). It can't be skipped.
        That's done intentionally, to keep signature monomorphic for high speed.

        """
    if self.__cache__ is None:
        self.__compile__()
        assert self.__cache__ is not None
    return self.__cache__.get(chainName, []) or []