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
def construct_d_case_6(num, den, x):
    s_inf = limit_at_inf(Poly(x ** 2, x) * num, den, x)
    if s_inf != -S(1) / 4:
        return [[(1 + sqrt(1 + 4 * s_inf)) / 2], [(1 - sqrt(1 + 4 * s_inf)) / 2]]
    return [[S.Half]]