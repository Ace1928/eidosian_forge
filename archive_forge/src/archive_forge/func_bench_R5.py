from sympy.core.random import random
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import factor
from sympy.simplify.simplify import simplify
from sympy.abc import x, y, z
from timeit import default_timer as clock
def bench_R5():
    """blowup(L, 8); L=uniq(L)"""

    def blowup(L, n):
        for i in range(n):
            L.append((L[i] + L[i + 1]) * L[i + 2])

    def uniq(x):
        v = set(x)
        return v
    L = [x, y, z]
    blowup(L, 8)
    L = uniq(L)