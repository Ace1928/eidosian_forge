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
def construct_d_case_5(ser):
    dplus = [0, 0]
    dplus[0] = sqrt(ser[0])
    dplus[-1] = ser[-1] / (2 * dplus[0])
    dminus = [-x for x in dplus]
    if dplus != dminus:
        return [dplus, dminus]
    return dplus