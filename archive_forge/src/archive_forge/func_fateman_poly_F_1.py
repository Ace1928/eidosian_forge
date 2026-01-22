from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent
from sympy.polys.rings import ring
def fateman_poly_F_1(n):
    """Fateman's GCD benchmark: trivial GCD """
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]
    y_0, y_1 = (Y[0], Y[1])
    u = y_0 + Add(*Y[1:])
    v = y_0 ** 2 + Add(*[y ** 2 for y in Y[1:]])
    F = ((u + 1) * (u + 2)).as_poly(*Y)
    G = ((v + 1) * (-3 * y_1 * y_0 ** 2 + y_1 ** 2 - 1)).as_poly(*Y)
    H = Poly(1, *Y)
    return (F, G, H)