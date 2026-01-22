from functools import wraps
from sympy.utilities.decorator import threaded, xthreaded, memoize_property, deprecated
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.core.basic import Basic
from sympy.core.relational import Eq
from sympy.matrices.dense import Matrix
from sympy.abc import x, y
def test_threaded():

    @threaded
    def function(expr, *args):
        return 2 * expr + sum(args)
    assert function(Matrix([[x, y], [1, x]]), 1, 2) == Matrix([[2 * x + 3, 2 * y + 3], [5, 2 * x + 3]])
    assert function(Eq(x, y), 1, 2) == Eq(2 * x + 3, 2 * y + 3)
    assert function([x, y], 1, 2) == [2 * x + 3, 2 * y + 3]
    assert function((x, y), 1, 2) == (2 * x + 3, 2 * y + 3)
    assert function({x, y}, 1, 2) == {2 * x + 3, 2 * y + 3}

    @threaded
    def function(expr, n):
        return expr ** n
    assert function(x + y, 2) == x ** 2 + y ** 2
    assert function(x, 2) == x ** 2