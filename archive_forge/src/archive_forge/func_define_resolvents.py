from sympy.core.evalf import (
from sympy.core.symbol import symbols, Dummy
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import ZZ
from sympy.polys.numberfields.resolvent_lookup import resolvent_coeff_lambdas
from sympy.polys.orderings import lex
from sympy.polys.polyroots import preprocess_roots
from sympy.polys.polytools import Poly
from sympy.polys.rings import xring
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.lambdify import lambdify
from mpmath import MPContext
from mpmath.libmp.libmpf import prec_to_dps
def define_resolvents():
    """Define all the resolvents for polys T of degree 4 through 6. """
    from sympy.combinatorics.galois import PGL2F5
    from sympy.combinatorics.permutations import Permutation
    R4, X4 = xring('X0,X1,X2,X3', ZZ, lex)
    X = X4
    F40 = X[0] * X[1] ** 2 + X[1] * X[2] ** 2 + X[2] * X[3] ** 2 + X[3] * X[0] ** 2
    s40 = [Permutation(3), Permutation(3)(0, 1), Permutation(3)(0, 2), Permutation(3)(0, 3), Permutation(3)(1, 2), Permutation(3)(2, 3)]
    F41 = X[0] * X[2] + X[1] * X[3]
    s41 = [Permutation(3), Permutation(3)(0, 1), Permutation(3)(0, 3)]
    R5, X5 = xring('X0,X1,X2,X3,X4', ZZ, lex)
    X = X5
    F51 = X[0] ** 2 * (X[1] * X[4] + X[2] * X[3]) + X[1] ** 2 * (X[2] * X[0] + X[3] * X[4]) + X[2] ** 2 * (X[3] * X[1] + X[4] * X[0]) + X[3] ** 2 * (X[4] * X[2] + X[0] * X[1]) + X[4] ** 2 * (X[0] * X[3] + X[1] * X[2])
    s51 = [Permutation(4), Permutation(4)(0, 1), Permutation(4)(0, 2), Permutation(4)(0, 3), Permutation(4)(0, 4), Permutation(4)(1, 4)]
    R6, X6 = xring('X0,X1,X2,X3,X4,X5', ZZ, lex)
    X = X6
    H = PGL2F5()
    term0 = X[0] ** 2 * X[5] ** 2 * (X[1] * X[4] + X[2] * X[3])
    terms = {term0.compose(list(zip(X, s(X)))) for s in H.elements}
    F61 = sum(terms)
    s61 = [Permutation(5)] + [Permutation(5)(0, n) for n in range(1, 6)]
    F62 = X[0] * X[1] * X[2] + X[3] * X[4] * X[5]
    s62 = [Permutation(5)] + [Permutation(5)(i, j + 3) for i in range(3) for j in range(3)]
    return {(4, 0): (F40, X4, s40), (4, 1): (F41, X4, s41), (5, 1): (F51, X5, s51), (6, 1): (F61, X6, s61), (6, 2): (F62, X6, s62)}