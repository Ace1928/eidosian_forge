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
def test_locate_joint_pos():
    N, A, P, C = _generate_body()
    joint = PinJoint('J', P, C, parent_point=N.y + N.z)
    assert joint.parent_point.name == 'J_P_joint'
    assert joint.parent_point.pos_from(P.masscenter) == N.y + N.z
    assert joint.child_point == C.masscenter
    N, A, P, C = _generate_body()
    parent_point = P.masscenter.locatenew('p', N.y + N.z)
    joint = PinJoint('J', P, C, parent_point=parent_point, child_point=C.masscenter)
    assert joint.parent_point == parent_point
    assert joint.child_point == C.masscenter
    N, A, P, C = _generate_body()
    raises(TypeError, lambda: PinJoint('J', P, C, parent_point=N.x.to_matrix(N)))
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    raises(ValueError, lambda: PinJoint('J', P, C, parent_point=q * N.x))
    N, A, P, C = _generate_body()
    child_point = C.masscenter.locatenew('p', q * A.y)
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))
    child_point = Point('p')
    raises(ValueError, lambda: PinJoint('J', P, C, child_point=child_point))