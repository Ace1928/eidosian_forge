from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix
def test_solve_lin_sys_2x4_none():
    domain, x1, x2 = ring('x1,x2', QQ)
    eqs = [x1 - 1, x1 - x2, x1 - 2 * x2, x2 - 1]
    assert solve_lin_sys(eqs, domain) is None