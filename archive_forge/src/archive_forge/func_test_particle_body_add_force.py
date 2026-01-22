from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_particle_body_add_force():
    particle_masscenter = Point('particle_masscenter')
    particle_mass = Symbol('particle_mass')
    particle_frame = ReferenceFrame('particle_frame')
    particle_body = Body('particle_body', particle_masscenter, particle_mass, particle_frame)
    a = Symbol('a')
    force_vector = a * particle_body.frame.x
    particle_body.apply_force(force_vector, particle_body.masscenter)
    assert len(particle_body.loads) == 1
    point = particle_body.masscenter.locatenew(particle_body._name + '_point0', 0)
    point.set_vel(particle_body.frame, 0)
    force_point = particle_body.loads[0][0]
    frame = particle_body.frame
    assert force_point.vel(frame) == point.vel(frame)
    assert force_point.pos_from(force_point) == point.pos_from(force_point)
    assert particle_body.loads[0][1] == force_vector