from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.numbers import igcdex, Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit
@cacheit
def cos_5() -> Expr:
    """Computes $\\cos \\frac{\\pi}{5}$ in square roots"""
    return (sqrt(5) + 1) / 4