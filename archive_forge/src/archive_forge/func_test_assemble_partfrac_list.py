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
def test_assemble_partfrac_list():
    f = 36 / (x ** 5 - 2 * x ** 4 - 2 * x ** 3 + 4 * x ** 2 + x - 2)
    pfd = apart_list(f)
    assert assemble_partfrac_list(pfd) == -4 / (x + 1) - 3 / (x + 1) ** 2 - 9 / (x - 1) ** 2 + 4 / (x - 2)
    a = Dummy('a')
    pfd = (1, Poly(0, x, domain='ZZ'), [([sqrt(2), -sqrt(2)], Lambda(a, a / 2), Lambda(a, -a + x), 1)])
    assert assemble_partfrac_list(pfd) == -1 / (sqrt(2) * (x + sqrt(2))) + 1 / (sqrt(2) * (x - sqrt(2)))