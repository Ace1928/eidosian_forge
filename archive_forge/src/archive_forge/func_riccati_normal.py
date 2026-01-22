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
def riccati_normal(w, x, b1, b2):
    """
    Given a solution `w(x)` to the equation

    .. math:: w'(x) = b_0(x) + b_1(x)*w(x) + b_2(x)*w(x)^2

    and rational function coefficients `b_1(x)` and
    `b_2(x)`, this function transforms the solution to
    give a solution `y(x)` for its corresponding normal
    Riccati ODE

    .. math:: y'(x) + y(x)^2 = a(x)

    using the transformation

    .. math:: y(x) = -b_2(x)*w(x) - b'_2(x)/(2*b_2(x)) - b_1(x)/2
    """
    return -b2 * w - b2.diff(x) / (2 * b2) - b1 / 2