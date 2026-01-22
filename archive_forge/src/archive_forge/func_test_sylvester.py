from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_sylvester():
    x = var('x')
    assert sylvester(x ** 3 - 7, 0, x) == sylvester(x ** 3 - 7, 0, x, 1) == Matrix([[0]])
    assert sylvester(0, x ** 3 - 7, x) == sylvester(0, x ** 3 - 7, x, 1) == Matrix([[0]])
    assert sylvester(x ** 3 - 7, 0, x, 2) == Matrix([[0]])
    assert sylvester(0, x ** 3 - 7, x, 2) == Matrix([[0]])
    assert sylvester(x ** 3 - 7, 7, x).det() == sylvester(x ** 3 - 7, 7, x, 1).det() == 343
    assert sylvester(7, x ** 3 - 7, x).det() == sylvester(7, x ** 3 - 7, x, 1).det() == 343
    assert sylvester(x ** 3 - 7, 7, x, 2).det() == -343
    assert sylvester(7, x ** 3 - 7, x, 2).det() == 343
    assert sylvester(3, 7, x).det() == sylvester(3, 7, x, 1).det() == sylvester(3, 7, x, 2).det() == 1
    assert sylvester(3, 0, x).det() == sylvester(3, 0, x, 1).det() == sylvester(3, 0, x, 2).det() == 1
    assert sylvester(x - 3, x - 8, x) == sylvester(x - 3, x - 8, x, 1) == sylvester(x - 3, x - 8, x, 2) == Matrix([[1, -3], [1, -8]])
    assert sylvester(x ** 3 - 7 * x + 7, 3 * x ** 2 - 7, x) == sylvester(x ** 3 - 7 * x + 7, 3 * x ** 2 - 7, x, 1) == Matrix([[1, 0, -7, 7, 0], [0, 1, 0, -7, 7], [3, 0, -7, 0, 0], [0, 3, 0, -7, 0], [0, 0, 3, 0, -7]])
    assert sylvester(x ** 3 - 7 * x + 7, 3 * x ** 2 - 7, x, 2) == Matrix([[1, 0, -7, 7, 0, 0], [0, 3, 0, -7, 0, 0], [0, 1, 0, -7, 7, 0], [0, 0, 3, 0, -7, 0], [0, 0, 1, 0, -7, 7], [0, 0, 0, 3, 0, -7]])