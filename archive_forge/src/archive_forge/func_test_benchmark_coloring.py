from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_benchmark_coloring():
    V = range(1, 12 + 1)
    E = [(1, 2), (2, 3), (1, 4), (1, 6), (1, 12), (2, 5), (2, 7), (3, 8), (3, 10), (4, 11), (4, 9), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (5, 12), (5, 9), (6, 10), (7, 11), (8, 12), (3, 4)]
    R, V = xring(['x%d' % v for v in V], QQ, lex)
    E = [(V[i - 1], V[j - 1]) for i, j in E]
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = V
    I3 = [x ** 3 - 1 for x in V]
    Ig = [x ** 2 + x * y + y ** 2 for x, y in E]
    I = I3 + Ig
    assert groebner(I[:-1], R) == [x1 + x11 + x12, x2 - x11, x3 - x12, x4 - x12, x5 + x11 + x12, x6 - x11, x7 - x12, x8 + x11 + x12, x9 - x11, x10 + x11 + x12, x11 ** 2 + x11 * x12 + x12 ** 2, x12 ** 3 - 1]
    assert groebner(I, R) == [1]