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
def construct_c_case_2(num, den, x, pole, mul):
    ri = mul // 2
    ser = rational_laurent_series(num, den, x, pole, mul, 6)
    cplus = [0 for i in range(ri)]
    cplus[ri - 1] = sqrt(ser[2 * ri])
    s = ri - 1
    sm = 0
    for s in range(ri - 1, 0, -1):
        sm = 0
        for j in range(s + 1, ri):
            sm += cplus[j - 1] * cplus[ri + s - j - 1]
        if s != 1:
            cplus[s - 1] = (ser[ri + s] - sm) / (2 * cplus[ri - 1])
    cminus = [-x for x in cplus]
    cplus[0] = (ser[ri + s] - sm - ri * cplus[ri - 1]) / (2 * cplus[ri - 1])
    cminus[0] = (ser[ri + s] - sm - ri * cminus[ri - 1]) / (2 * cminus[ri - 1])
    if cplus != cminus:
        return [cplus, cminus]
    return cplus