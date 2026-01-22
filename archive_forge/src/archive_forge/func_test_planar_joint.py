from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core.backend import Matrix, _simplify_matrix, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (dynamicsymbols, Body, JointsMethod,
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_planar_joint():
    N, A, P, C = _generate_body()
    q0_def, q1_def, q2_def = dynamicsymbols('q0:3_J')
    u0_def, u1_def, u2_def = dynamicsymbols('u0:3_J')
    Cj = PlanarJoint('J', P, C)
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def, q2_def])
    assert Cj.speeds == Matrix([u0_def, u1_def, u2_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.planar_coordinates == Matrix([q1_def, q2_def])
    assert Cj.rotation_speed == u0_def
    assert Cj.planar_speeds == Matrix([u1_def, u2_def])
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t), u2_def - q2_def.diff(t)])
    assert Cj.rotation_axis == N.x
    assert Cj.planar_vectors == [N.y, N.z]
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)
    r_P_C = q1_def * N.y + q2_def * N.z
    assert Cj.parent_point.pos_from(Cj.child_point) == -r_P_C
    assert C.masscenter.pos_from(P.masscenter) == r_P_C
    assert Cj.child_point.vel(N) == u1_def * N.y + u2_def * N.z
    assert A.ang_vel_in(N) == u0_def * N.x
    assert Cj.parent_interframe == N
    assert Cj.child_interframe == A
    assert Cj.__str__() == 'PlanarJoint: J  parent: P  child: C'
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    l, m = symbols('l, m')
    N, A, P, C, Pint, Cint = _generate_body(True)
    Cj = PlanarJoint('J', P, C, rotation_coordinate=q0, planar_coordinates=[q1, q2], planar_speeds=[u1, u2], parent_point=m * N.x, child_point=l * A.y, parent_interframe=Pint, child_interframe=Cint)
    assert Cj.coordinates == Matrix([q0, q1, q2])
    assert Cj.speeds == Matrix([u0_def, u1, u2])
    assert Cj.rotation_coordinate == q0
    assert Cj.planar_coordinates == Matrix([q1, q2])
    assert Cj.rotation_speed == u0_def
    assert Cj.planar_speeds == Matrix([u1, u2])
    assert Cj.kdes == Matrix([u0_def - q0.diff(t), u1 - q1.diff(t), u2 - q2.diff(t)])
    assert Cj.rotation_axis == Pint.x
    assert Cj.planar_vectors == [Pint.y, Pint.z]
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x
    assert Cj.parent_point.pos_from(Cj.child_point) == q1 * N.y + q2 * N.z
    assert C.masscenter.pos_from(P.masscenter) == m * N.x - q1 * N.y - q2 * N.z - l * A.y
    assert C.masscenter.vel(N) == -u1 * N.y - u2 * N.z + u0_def * l * A.x
    assert A.ang_vel_in(N) == u0_def * N.x