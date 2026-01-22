from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_res_z():
    x = var('x')
    assert res_z(3, 5, x) == 1
    assert res(3, 5, x) == res_q(3, 5, x) == res_z(3, 5, x)