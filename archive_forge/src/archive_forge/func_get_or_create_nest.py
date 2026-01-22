from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def get_or_create_nest(self, key: Key, *, access_lists: bool=True) -> dict:
    cont: Any = self.dict
    for k in key:
        if k not in cont:
            cont[k] = {}
        cont = cont[k]
        if access_lists and isinstance(cont, list):
            cont = cont[-1]
        if not isinstance(cont, dict):
            raise KeyError('There is no nest behind this key')
    return cont