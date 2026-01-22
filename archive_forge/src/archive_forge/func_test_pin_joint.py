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
def test_pin_joint():
    P = Body('P')
    C = Body('C')
    l, m = symbols('l m')
    q, u = dynamicsymbols('q_J, u_J')
    Pj = PinJoint('J', P, C)
    assert Pj.name == 'J'
    assert Pj.parent == P
    assert Pj.child == C
    assert Pj.coordinates == Matrix([q])
    assert Pj.speeds == Matrix([u])
    assert Pj.kdes == Matrix([u - q.diff(t)])
    assert Pj.joint_axis == P.frame.x
    assert Pj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Pj.parent_point.pos_from(Pj._child_point) == Vector(0)
    assert C.masscenter.pos_from(P.masscenter) == Vector(0)
    assert Pj.parent_interframe == P.frame
    assert Pj.child_interframe == C.frame
    assert Pj.__str__() == 'PinJoint: J  parent: P  child: C'
    P1 = Body('P1')
    C1 = Body('C1')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P1.frame, P1.y, pi / 2)
    J1 = PinJoint('J1', P1, C1, parent_point=l * P1.frame.x, child_point=m * C1.frame.y, joint_axis=P1.frame.z, parent_interframe=Pint)
    assert J1._joint_axis == P1.frame.z
    assert J1._child_point.pos_from(C1.masscenter) == m * C1.frame.y
    assert J1._parent_point.pos_from(P1.masscenter) == l * P1.frame.x
    assert J1._parent_point.pos_from(J1._child_point) == Vector(0)
    assert P1.masscenter.pos_from(C1.masscenter) == -l * P1.frame.x + m * C1.frame.y
    assert J1.parent_interframe == Pint
    assert J1.child_interframe == C1.frame
    q, u = dynamicsymbols('q, u')
    N, A, P, C, Pint, Cint = _generate_body(True)
    parent_point = P.masscenter.locatenew('parent_point', N.x + N.y)
    child_point = C.masscenter.locatenew('child_point', C.y + C.z)
    J = PinJoint('J', P, C, q, u, parent_point=parent_point, child_point=child_point, parent_interframe=Pint, child_interframe=Cint, joint_axis=N.z)
    assert J.joint_axis == N.z
    assert J.parent_point.vel(N) == 0
    assert J.parent_point == parent_point
    assert J.child_point == child_point
    assert J.child_point.pos_from(P.masscenter) == N.x + N.y
    assert J.parent_point.pos_from(C.masscenter) == C.y + C.z
    assert C.masscenter.pos_from(P.masscenter) == N.x + N.y - C.y - C.z
    assert C.masscenter.vel(N).express(N) == (u * sin(q) - u * cos(q)) * N.x + (-u * sin(q) - u * cos(q)) * N.y
    assert J.parent_interframe == Pint
    assert J.child_interframe == Cint