from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_subresultants_bezout():
    x = var('x')
    p = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    q = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    assert subresultants_bezout(p, q, x) == subresultants(p, q, x)
    assert subresultants_bezout(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_bezout(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_bezout(p, q, x) == [i * j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]
    p = x ** 3 - 7 * x + 7
    q = 3 * x ** 2 - 7
    assert subresultants_bezout(p, q, x) == euclid_amv(p, q, x)