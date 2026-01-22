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
def bench_R10():
    """v = [-pi,-pi+1/10..,pi]"""

    def srange(min, max, step):
        v = [min]
        while (max - v[-1]).evalf() > 0:
            v.append(v[-1] + step)
        return v[:-1]
    srange(-pi, pi, sympify(1) / 10)