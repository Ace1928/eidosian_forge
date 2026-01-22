from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
from io import StringIO
def rl1(x: int) -> int:
    if x == 1:
        return 2
    return x