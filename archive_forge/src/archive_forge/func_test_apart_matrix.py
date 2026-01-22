from sympy.polys.partfrac import (
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x, y, a, b, c
def test_apart_matrix():
    M = Matrix(2, 2, lambda i, j: 1 / (x + i + 1) / (x + j))
    assert apart(M) == Matrix([[1 / x - 1 / (x + 1), (x + 1) ** (-2)], [1 / (2 * x) - S.Half / (x + 2), 1 / (x + 1) - 1 / (x + 2)]])