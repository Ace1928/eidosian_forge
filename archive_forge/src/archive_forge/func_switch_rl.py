from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def switch_rl(expr: _S) -> _S:
    rl = ruledict.get(key(expr), identity)
    return rl(expr)