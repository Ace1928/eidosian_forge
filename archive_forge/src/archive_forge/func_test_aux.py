from sympy import solve
from sympy.core.backend import (cos, expand, Matrix, sin, symbols, tan, sqrt, S,
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.testing.pytest import raises
from sympy.core.backend import USE_SYMENGINE
def test_aux():
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    q1d, q2d, q3d, u1d, u2d, u3d = dynamicsymbols('q1 q2 q3 u1 u2 u3', 1)
    u4, u5, f1, f2 = dynamicsymbols('u4, u5, f1, f2')
    u4d, u5d = dynamicsymbols('u4, u5', 1)
    r, m, g = symbols('r m g')
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    w_R_N_qd = R.ang_vel_in(N)
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)
    C = Point('C')
    C.set_vel(N, u4 * L.x + u5 * (Y.z ^ L.x))
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)
    Dmc.a2pt_theory(C, N, R)
    I = inertia(L, m / 4 * r ** 2, m / 2 * r ** 2, m / 4 * r ** 2)
    kd = [dot(R.ang_vel_in(N) - w_R_N_qd, uv) for uv in L]
    ForceList = [(Dmc, -m * g * Y.z), (C, f1 * L.x + f2 * (Y.z ^ L.x))]
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyList = [BodyD]
    KM = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3, u4, u5], kd_eqs=kd)
    fr, frstar = KM.kanes_equations(BodyList, ForceList)
    fr = fr.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    frstar = frstar.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    KM2 = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3], kd_eqs=kd, u_auxiliary=[u4, u5])
    fr2, frstar2 = KM2.kanes_equations(BodyList, ForceList)
    fr2 = fr2.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    frstar2 = frstar2.subs({u4d: 0, u5d: 0}).subs({u4: 0, u5: 0})
    frstar.simplify()
    frstar2.simplify()
    assert (fr - fr2).expand() == Matrix([0, 0, 0, 0, 0])
    assert (frstar - frstar2).expand() == Matrix([0, 0, 0, 0, 0])