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
def test_locate_joint_frame():
    N, A, P, C = _generate_body()
    parent_interframe = ReferenceFrame('int_frame')
    parent_interframe.orient_axis(N, N.z, 1)
    joint = PinJoint('J', P, C, parent_interframe=parent_interframe)
    assert joint.parent_interframe == parent_interframe
    assert joint.parent_interframe.ang_vel_in(N) == 0
    assert joint.child_interframe == A
    q = dynamicsymbols('q')
    N, A, P, C = _generate_body()
    parent_interframe = ReferenceFrame('int_frame')
    parent_interframe.orient_axis(N, N.z, q)
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=parent_interframe))
    N, A, P, C = _generate_body()
    child_interframe = ReferenceFrame('int_frame')
    child_interframe.orient_axis(N, N.z, 1)
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=child_interframe))