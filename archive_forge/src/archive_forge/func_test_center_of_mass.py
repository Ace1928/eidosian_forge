from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_center_of_mass():
    a = ReferenceFrame('a')
    m = symbols('m', real=True)
    p1 = Particle('p1', Point('p1_pt'), S.One)
    p2 = Particle('p2', Point('p2_pt'), S(2))
    p3 = Particle('p3', Point('p3_pt'), S(3))
    p4 = Particle('p4', Point('p4_pt'), m)
    b_f = ReferenceFrame('b_f')
    b_cm = Point('b_cm')
    mb = symbols('mb')
    b = RigidBody('b', b_cm, b_f, mb, (outer(b_f.x, b_f.x), b_cm))
    p2.point.set_pos(p1.point, a.x)
    p3.point.set_pos(p1.point, a.x + a.y)
    p4.point.set_pos(p1.point, a.y)
    b.masscenter.set_pos(p1.point, a.y + a.z)
    point_o = Point('o')
    point_o.set_pos(p1.point, center_of_mass(p1.point, p1, p2, p3, p4, b))
    expr = 5 / (m + mb + 6) * a.x + (m + mb + 3) / (m + mb + 6) * a.y + mb / (m + mb + 6) * a.z
    assert point_o.pos_from(p1.point) - expr == 0