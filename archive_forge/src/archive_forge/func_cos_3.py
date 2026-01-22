from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.numbers import igcdex, Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit
@cacheit
def cos_3() -> Expr:
    """Computes $\\cos \\frac{\\pi}{3}$ in square roots"""
    return S.Half