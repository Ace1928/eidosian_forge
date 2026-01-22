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
def fateman_poly_F_3(n):
    """Fateman's GCD benchmark: sparse inputs (deg f ~ vars f) """
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]
    y_0 = Y[0]
    u = Add(*[y ** (n + 1) for y in Y[1:]])
    H = Poly((y_0 ** (n + 1) + u + 1) ** 2, *Y)
    F = Poly((y_0 ** (n + 1) - u - 2) ** 2, *Y)
    G = Poly((y_0 ** (n + 1) + u + 2) ** 2, *Y)
    return (H * F, H * G, H)