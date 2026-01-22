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
def bench_R8():
    """right(x^2,0,5,10^4)"""

    def right(f, a, b, n):
        a = sympify(a)
        b = sympify(b)
        n = sympify(n)
        x = f.atoms(Symbol).pop()
        Deltax = (b - a) / n
        c = a
        est = 0
        for i in range(n):
            c += Deltax
            est += f.subs(x, c)
        return est * Deltax
    right(x ** 2, 0, 5, 10 ** 4)