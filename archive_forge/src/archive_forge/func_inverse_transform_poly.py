from itertools import product
from sympy.core import S
from sympy.core.add import Add
from sympy.core.numbers import oo, Float
from sympy.core.function import count_ops
from sympy.core.relational import Eq
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions import sqrt, exp
from sympy.functions.elementary.complexes import sign
from sympy.integrals.integrals import Integral
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyroots import roots
from sympy.solvers.solveset import linsolve
def inverse_transform_poly(num, den, x):
    """
    A function to make the substitution
    x -> 1/x in a rational function that
    is represented using Poly objects for
    numerator and denominator.
    """
    one = Poly(1, x)
    xpoly = Poly(x, x)
    pwr = val_at_inf(num, den, x)
    if pwr >= 0:
        if num.expr != 0:
            num = num.transform(one, xpoly) * x ** pwr
            den = den.transform(one, xpoly)
    else:
        num = num.transform(one, xpoly)
        den = den.transform(one, xpoly) * x ** (-pwr)
    return num.cancel(den, include=True)