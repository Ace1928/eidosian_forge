from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_body_add_force():
    rigidbody_masscenter = Point('rigidbody_masscenter')
    rigidbody_mass = Symbol('rigidbody_mass')
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    rigid_body = Body('rigidbody_body', rigidbody_masscenter, rigidbody_mass, rigidbody_frame, body_inertia)
    l = Symbol('l')
    Fa = Symbol('Fa')
    point = rigid_body.masscenter.locatenew('rigidbody_body_point0', l * rigid_body.frame.x)
    point.set_vel(rigid_body.frame, 0)
    force_vector = Fa * rigid_body.frame.z
    rigid_body.apply_force(force_vector, point)
    assert len(rigid_body.loads) == 1
    force_point = rigid_body.loads[0][0]
    frame = rigid_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    assert force_point.pos_from(force_point) == point.pos_from(force_point)
    assert rigid_body.loads[0][1] == force_vector
    rigid_body.apply_force(force_vector)
    assert len(rigid_body.loads) == 2
    assert rigid_body.loads[1][1] == force_vector
    raises(TypeError, lambda: rigid_body.apply_force(force_vector, 0))
    raises(TypeError, lambda: rigid_body.apply_force(0))