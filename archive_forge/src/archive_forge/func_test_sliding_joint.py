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
def test_sliding_joint():
    _, _, P, C = _generate_body()
    q, u = dynamicsymbols('q_S, u_S')
    S = PrismaticJoint('S', P, C)
    assert S.name == 'S'
    assert S.parent == P
    assert S.child == C
    assert S.coordinates == Matrix([q])
    assert S.speeds == Matrix([u])
    assert S.kdes == Matrix([u - q.diff(t)])
    assert S.joint_axis == P.frame.x
    assert S.child_point.pos_from(C.masscenter) == Vector(0)
    assert S.parent_point.pos_from(P.masscenter) == Vector(0)
    assert S.parent_point.pos_from(S.child_point) == -q * P.frame.x
    assert P.masscenter.pos_from(C.masscenter) == -q * P.frame.x
    assert C.masscenter.vel(P.frame) == u * P.frame.x
    assert P.ang_vel_in(C) == 0
    assert C.ang_vel_in(P) == 0
    assert S.__str__() == 'PrismaticJoint: S  parent: P  child: C'
    N, A, P, C = _generate_body()
    l, m = symbols('l m')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.x, child_point=m * C.frame.y, joint_axis=P.frame.z, parent_interframe=Pint)
    assert S.joint_axis == P.frame.z
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.y
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.x
    assert S.parent_point.pos_from(S.child_point) == -q * P.frame.z
    assert P.masscenter.pos_from(C.masscenter) == -l * N.x - q * N.z + m * A.y
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.ang_vel_in(P) == 0
    assert P.ang_vel_in(C) == 0
    _, _, P, C = _generate_body()
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    S = PrismaticJoint('S', P, C, parent_point=l * P.frame.z, child_point=m * C.frame.x, joint_axis=P.frame.z, parent_interframe=Pint)
    assert S.joint_axis == P.frame.z
    assert S.child_point.pos_from(C.masscenter) == m * C.frame.x
    assert S.parent_point.pos_from(P.masscenter) == l * P.frame.z
    assert S.parent_point.pos_from(S.child_point) == -q * P.frame.z
    assert P.masscenter.pos_from(C.masscenter) == (-l - q) * P.frame.z + m * C.frame.x
    assert C.masscenter.vel(P.frame) == u * P.frame.z
    assert C.ang_vel_in(P) == 0
    assert P.ang_vel_in(C) == 0