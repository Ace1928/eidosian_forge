from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def null_safe_rl(expr: _T) -> _T:
    result = rule(expr)
    if result is None:
        return expr
    return result