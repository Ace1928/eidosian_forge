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
def construct_d(num, den, x, val_inf):
    """
    Helper function to calculate the coefficients
    in the d-vector based on the valuation of the
    function at oo.
    """
    N = -val_inf // 2
    mul = -val_inf if val_inf < 0 else 0
    ser = rational_laurent_series(num, den, x, oo, mul, 1)
    if val_inf < 0:
        d = construct_d_case_4(ser, N)
    elif val_inf == 0:
        d = construct_d_case_5(ser)
    else:
        d = construct_d_case_6(num, den, x)
    return d