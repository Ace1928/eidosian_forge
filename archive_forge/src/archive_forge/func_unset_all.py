from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def unset_all(self, key: Key) -> None:
    cont = self._flags
    for k in key[:-1]:
        if k not in cont:
            return
        cont = cont[k]['nested']
    cont.pop(key[-1], None)