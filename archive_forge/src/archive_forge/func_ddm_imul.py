from __future__ import annotations
from operator import mul
from .exceptions import (
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement
def ddm_imul(a: list[list[R]], b: R) -> None:
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = aij * b