from sympy.core.backend import (symbols, Matrix, cos, sin, atan, sqrt,
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,\
from sympy.testing.pytest import slow
def test_linearize_rolling_disc_lagrange():
    q1, q2, q3 = q = dynamicsymbols('q1 q2 q3')
    q1d, q2d, q3d = qd = dynamicsymbols('q1 q2 q3', 1)
    r, m, g = symbols('r m g')
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    C = Point('C')
    C.set_vel(N, 0)
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)
    I = inertia(L, m / 4 * r ** 2, m / 2 * r ** 2, m / 4 * r ** 2)
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyD.potential_energy = -m * g * r * cos(q2)
    Lag = Lagrangian(N, BodyD)
    l = LagrangesMethod(Lag, q)
    l.form_lagranges_equations()
    op_point = {q1: 0, q2: 0, q3: 0, q1d: 0, q2d: 0, q1d.diff(): 0, q2d.diff(): 0, q3d.diff(): 0}
    A = l.linearize(q_ind=q, qd_ind=qd, op_point=op_point, A_and_B=True)[0]
    sol = Matrix([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -6 * q3d, 0], [0, -4 * g / (5 * r), 0, 6 * q3d / 5, 0, 0], [0, 0, 0, 0, 0, 0]])
    assert A == sol