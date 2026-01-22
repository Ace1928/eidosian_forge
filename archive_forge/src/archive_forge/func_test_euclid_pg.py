from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_euclid_pg():
    x = var('x')
    p = x ** 6 + x ** 5 - x ** 4 - x ** 3 + x ** 2 - x + 1
    q = 6 * x ** 5 + 5 * x ** 4 - 4 * x ** 3 - 3 * x ** 2 + 2 * x - 1
    assert euclid_pg(p, q, x)[-1] == sylvester(p, q, x).det()
    assert euclid_pg(p, q, x) == subresultants_pg(p, q, x)
    p = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    q = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    assert euclid_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    sam_factors = [1, 1, -1, -1, 1, 1]
    assert euclid_pg(p, q, x) == [i * j for i, j in zip(sam_factors, sturm_pg(p, q, x))]