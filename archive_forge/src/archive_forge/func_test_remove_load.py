from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_remove_load():
    P1 = Point('P1')
    P2 = Point('P2')
    B = Body('B')
    f1 = B.x
    f2 = B.y
    B.apply_force(f1, P1)
    B.apply_force(f2, P2)
    assert B.loads == [(P1, f1), (P2, f2)]
    B.remove_load(P2)
    assert B.loads == [(P1, f1)]
    B.apply_torque(f1.cross(f2))
    assert B.loads == [(P1, f1), (B.frame, f1.cross(f2))]
    B.remove_load()
    assert B.loads == [(P1, f1)]