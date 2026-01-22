from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_apply_loads_on_multi_degree_freedom_holonomic_system():
    """Example based on: https://pydy.readthedocs.io/en/latest/examples/multidof-holonomic.html"""
    W = Body('W')
    B = Body('B')
    P = Body('P')
    b = Body('b')
    q1, q2 = dynamicsymbols('q1 q2')
    k, c, g, kT = symbols('k c g kT')
    F, T = dynamicsymbols('F T')
    B.apply_force(F * W.x)
    W.apply_force(k * q1 * W.x, reaction_body=B)
    W.apply_force(c * q1.diff() * W.x, reaction_body=B)
    P.apply_force(P.mass * g * W.y)
    b.apply_force(b.mass * g * W.y)
    P.apply_torque(kT * q2 * W.z, reaction_body=b)
    P.apply_torque(T * W.z)
    assert B.loads == [(B.masscenter, (F - k * q1 - c * q1.diff()) * W.x)]
    assert P.loads == [(P.masscenter, P.mass * g * W.y), (P.frame, (T + kT * q2) * W.z)]
    assert b.loads == [(b.masscenter, b.mass * g * W.y), (b.frame, -kT * q2 * W.z)]
    assert W.loads == [(W.masscenter, (c * q1.diff() + k * q1) * W.x)]