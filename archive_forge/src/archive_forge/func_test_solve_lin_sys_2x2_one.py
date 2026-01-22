from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix
def test_solve_lin_sys_2x2_one():
    domain, x1, x2 = ring('x1,x2', QQ)
    eqs = [x1 + x2 - 5, 2 * x1 - x2]
    sol = {x1: QQ(5, 3), x2: QQ(10, 3)}
    _sol = solve_lin_sys(eqs, domain)
    assert _sol == sol and all((isinstance(s, domain.dtype) for s in _sol))