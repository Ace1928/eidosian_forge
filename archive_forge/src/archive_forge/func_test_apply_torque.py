from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_apply_torque():
    t = symbols('t')
    q = dynamicsymbols('q')
    B1 = Body('B1')
    B2 = Body('B2')
    N = ReferenceFrame('N')
    torque = t * q * N.x
    B1.apply_torque(torque, B2)
    assert B1.loads == [(B1.frame, torque)]
    assert B2.loads == [(B2.frame, -torque)]
    torque2 = t * N.y
    B1.apply_torque(torque2)
    assert B1.loads == [(B1.frame, torque + torque2)]