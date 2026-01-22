from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
def test_dub_pen():
    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    u1, u2 = dynamicsymbols('u1 u2')
    u1d, u2d = dynamicsymbols('u1 u2', 1)
    l, m, g = symbols('l m g')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = N.orientnew('B', 'Axis', [q2, N.z])
    A.set_ang_vel(N, q1d * A.z)
    B.set_ang_vel(N, q2d * A.z)
    O = Point('O')
    P = O.locatenew('P', l * A.x)
    R = P.locatenew('R', l * B.x)
    O.set_vel(N, 0)
    P.v2pt_theory(O, N, A)
    R.v2pt_theory(P, N, B)
    ParP = Particle('ParP', P, m)
    ParR = Particle('ParR', R, m)
    ParP.potential_energy = -m * g * l * cos(q1)
    ParR.potential_energy = -m * g * l * cos(q1) - m * g * l * cos(q2)
    L = Lagrangian(N, ParP, ParR)
    lm = LagrangesMethod(L, [q1, q2], bodies=[ParP, ParR])
    lm.form_lagranges_equations()
    assert simplify(l * m * (2 * g * sin(q1) + l * sin(q1) * sin(q2) * q2dd + l * sin(q1) * cos(q2) * q2d ** 2 - l * sin(q2) * cos(q1) * q2d ** 2 + l * cos(q1) * cos(q2) * q2dd + 2 * l * q1dd) - lm.eom[0]) == 0
    assert simplify(l * m * (g * sin(q2) + l * sin(q1) * sin(q2) * q1dd - l * sin(q1) * cos(q2) * q1d ** 2 + l * sin(q2) * cos(q1) * q1d ** 2 + l * cos(q1) * cos(q2) * q1dd + l * q2dd) - lm.eom[1]) == 0
    assert lm.bodies == [ParP, ParR]