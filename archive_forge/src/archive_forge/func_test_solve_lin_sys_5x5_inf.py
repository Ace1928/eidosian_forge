from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix
def test_solve_lin_sys_5x5_inf():
    domain, x1, x2, x3, x4, x5 = ring('x1,x2,x3,x4,x5', QQ)
    eqs = [x1 - x2 - 2 * x3 + x4 + 11 * x5 - 13, x1 - x2 + x3 + x4 + 5 * x5 - 16, 2 * x1 - 2 * x2 + x4 + 10 * x5 - 21, 2 * x1 - 2 * x2 - x3 + 3 * x4 + 20 * x5 - 38, 2 * x1 - 2 * x2 + x3 + x4 + 8 * x5 - 22]
    sol = {x1: 6 + x2 - 3 * x5, x3: 1 + 2 * x5, x4: 9 - 4 * x5}
    assert solve_lin_sys(eqs, domain) == sol