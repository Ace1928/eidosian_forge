from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_apply_force():
    f, g = symbols('f g')
    q, x, v1, v2 = dynamicsymbols('q x v1 v2')
    P1 = Point('P1')
    P2 = Point('P2')
    B1 = Body('B1')
    B2 = Body('B2')
    N = ReferenceFrame('N')
    P1.set_vel(B1.frame, v1 * B1.x)
    P2.set_vel(B2.frame, v2 * B2.x)
    force = f * q * N.z
    B1.apply_force(force, P1, B2, P2)
    assert B1.loads == [(P1, force)]
    assert B2.loads == [(P2, -force)]
    g1 = B1.mass * g * N.y
    g2 = B2.mass * g * N.y
    B1.apply_force(g1)
    B2.apply_force(g2)
    assert B1.loads == [(P1, force), (B1.masscenter, g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, g2)]
    force2 = x * N.x
    B1.apply_force(force2, reaction_body=B2)
    assert B1.loads == [(P1, force), (B1.masscenter, force2 + g1)]
    assert B2.loads == [(P2, -force), (B2.masscenter, -force2 + g2)]