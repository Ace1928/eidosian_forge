from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_sturm_pg():
    x = var('x')
    p = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    q = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    assert sturm_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    sam_factors = [1, 1, -1, -1, 1, 1]
    assert sturm_pg(p, q, x) == [i * j for i, j in zip(sam_factors, euclid_pg(p, q, x))]
    p = -9 * x ** 5 - 5 * x ** 3 - 9
    q = -45 * x ** 4 - 15 * x ** 2
    assert sturm_pg(p, q, x, 1)[-1] == sylvester(p, q, x, 1).det()
    assert sturm_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    assert sturm_pg(-p, q, x)[-1] == sylvester(-p, q, x, 2).det()
    assert sturm_pg(-p, q, x) == modified_subresultants_pg(-p, q, x)