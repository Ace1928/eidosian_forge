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
@XFAIL
def test_noncommutative_pseudomultivariate():

    class foo(Expr):
        is_commutative = False
    e = x / (x + x * y)
    c = 1 / (1 + y)
    assert apart(e + foo(e)) == c + foo(c)
    assert apart(e * foo(e)) == c * foo(c)