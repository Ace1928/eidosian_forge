from sympy import solve
from sympy.core.backend import (cos, expand, Matrix, sin, symbols, tan, sqrt, S,
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.testing.pytest import raises
from sympy.core.backend import USE_SYMENGINE
def test_one_dof():
    q, u = dynamicsymbols('q u')
    qd, ud = dynamicsymbols('q u', 1)
    m, c, k = symbols('m c k')
    N = ReferenceFrame('N')
    P = Point('P')
    P.set_vel(N, u * N.x)
    kd = [qd - u]
    FL = [(P, (-k * q - c * u) * N.x)]
    pa = Particle('pa', P, m)
    BL = [pa]
    KM = KanesMethod(N, [q], [u], kd)
    KM.kanes_equations(BL, FL)
    assert KM.bodies == BL
    assert KM.loads == FL
    MM = KM.mass_matrix
    forcing = KM.forcing
    rhs = MM.inv() * forcing
    assert expand(rhs[0]) == expand(-(q * k + u * c) / m)
    assert simplify(KM.rhs() - KM.mass_matrix_full.LUsolve(KM.forcing_full)) == zeros(2, 1)
    assert KM.linearize(A_and_B=True)[0] == Matrix([[0, 1], [-k / m, -c / m]])