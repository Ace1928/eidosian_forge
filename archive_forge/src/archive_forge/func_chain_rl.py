from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout
def chain_rl(expr: _T) -> _T:
    for rule in rules:
        expr = rule(expr)
    return expr