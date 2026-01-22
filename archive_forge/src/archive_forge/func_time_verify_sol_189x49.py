from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys
def time_verify_sol_189x49():
    eqs = eqs_189x49()
    sol = sol_189x49()
    zeros = [eq.compose(sol) for eq in eqs]
    assert all((zero == 0 for zero in zeros))