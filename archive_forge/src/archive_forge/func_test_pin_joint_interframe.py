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
def test_pin_joint_interframe():
    q, u = dynamicsymbols('q, u')
    N, A, P, C = _generate_body()
    Pint, Cint = (ReferenceFrame('Pint'), ReferenceFrame('Cint'))
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    Pint.orient_axis(N, N.z, q)
    Cint.orient_axis(A, A.z, q)
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint))
    raises(ValueError, lambda: PinJoint('J', P, C, child_interframe=Cint))
    N, A, P, C = _generate_body()
    Pint = ReferenceFrame('Pint')
    Pint.orient_body_fixed(N, (pi / 4, pi, pi / 3), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x, child_point=-C.y, parent_interframe=Pint, joint_axis=Pint.x)
    assert _simplify_matrix(N.dcm(A)) - Matrix([[-1 / 2, sqrt(3) * cos(q) / 2, -sqrt(3) * sin(q) / 2], [sqrt(6) / 4, sqrt(2) * (2 * sin(q) + cos(q)) / 4, sqrt(2) * (-sin(q) + 2 * cos(q)) / 4], [sqrt(6) / 4, sqrt(2) * (-2 * sin(q) + cos(q)) / 4, -sqrt(2) * (sin(q) + 2 * cos(q)) / 4]]) == zeros(3)
    assert A.ang_vel_in(N) == u * Pint.x
    assert C.masscenter.pos_from(P.masscenter) == N.x + A.y
    assert C.masscenter.vel(N) == u * A.z
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.masscenter.vel(Pint) == u * A.z
    N, A, P, C = _generate_body()
    Cint = ReferenceFrame('Cint')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=-N.z, child_point=C.x, child_interframe=Cint, joint_axis=P.x + P.z)
    assert _simplify_matrix(N.dcm(A)) == Matrix([[-sqrt(2) * sin(q) / 2, -sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4, sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4], [cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4, (-sqrt(2) + sqrt(6)) * sin(q) / 4], [sqrt(2) * sin(q) / 2, sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4, sqrt(3) * (1 - cos(q)) / 4 + cos(q) / 4 + S(1) / 4]])
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * N.x + sqrt(2) * u / 2 * N.z
    assert C.masscenter.pos_from(P.masscenter) == -N.z - A.x
    assert C.masscenter.vel(N).simplify() == (-sqrt(6) - sqrt(2)) * u / 4 * A.y + (-sqrt(2) + sqrt(6)) * u / 4 * A.z
    assert C.masscenter.vel(Cint) == Vector(0)
    N, A, P, C = _generate_body()
    Pint, Cint = (ReferenceFrame('Pint'), ReferenceFrame('Cint'))
    Pint.orient_body_fixed(N, (-pi / 2, pi, pi / 2), 'xyz')
    Cint.orient_body_fixed(A, (2 * pi / 3, -pi, pi / 2), 'xyz')
    PinJoint('J', P, C, q, u, parent_point=N.x - N.y, child_point=-C.z, parent_interframe=Pint, child_interframe=Cint, joint_axis=Pint.x + Pint.z)
    assert _simplify_matrix(N.dcm(A)) == Matrix([[cos(q), (sqrt(2) + sqrt(6)) * -sin(q) / 4, (-sqrt(2) + sqrt(6)) * sin(q) / 4], [-sqrt(2) * sin(q) / 2, -sqrt(3) * (cos(q) + 1) / 4 - cos(q) / 4 + S(1) / 4, sqrt(3) * (cos(q) - 1) / 4 - cos(q) / 4 - S(1) / 4], [sqrt(2) * sin(q) / 2, sqrt(3) * (cos(q) - 1) / 4 + cos(q) / 4 + S(1) / 4, -sqrt(3) * (cos(q) + 1) / 4 + cos(q) / 4 - S(1) / 4]])
    assert A.ang_vel_in(N) == sqrt(2) * u / 2 * Pint.x + sqrt(2) * u / 2 * Pint.z
    assert C.masscenter.pos_from(P.masscenter) == N.x - N.y + A.z
    N_v_C = (-sqrt(2) + sqrt(6)) * u / 4 * A.x
    assert C.masscenter.vel(N).simplify() == N_v_C
    assert C.masscenter.vel(Pint).simplify() == N_v_C
    assert C.masscenter.vel(Cint) == Vector(0)