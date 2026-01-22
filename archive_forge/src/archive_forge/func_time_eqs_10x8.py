from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys
def time_eqs_10x8():
    if len(eqs_10x8()) != 10:
        raise ValueError('Value should be equal to 10')