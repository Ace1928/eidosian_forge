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
def test_sliding_joint_arbitrary_axis():
    q, u = dynamicsymbols('q_S, u_S')
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, child_interframe=-A.x)
    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert C.masscenter.pos_from(P.masscenter) == q * N.x
    assert C.masscenter.pos_from(P.masscenter).express(A).simplify() == -q * A.x
    assert C.masscenter.vel(N) == u * N.x
    assert C.masscenter.vel(N).express(A) == -u * A.x
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, child_interframe=A.y, child_point=A.x)
    assert A.y.angle_between(N.x) == 0
    assert A.y.express(N) == N.x
    assert A.dcm(N) == Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert C.masscenter.vel(N) == u * N.x
    assert C.masscenter.vel(N).express(A) == u * A.y
    assert C.masscenter.pos_from(P.masscenter) == q * N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == q * N.x + N.y
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_interframe=N.y, parent_point=N.x)
    assert N.y.angle_between(A.x) == 0
    assert N.y.express(A) == A.x
    assert A.dcm(N) == Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert C.masscenter.vel(N) == u * N.y
    assert C.masscenter.vel(N).express(A) == u * A.x
    assert C.masscenter.pos_from(P.masscenter) == N.x + q * N.y
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x, child_interframe=A.x + A.y)
    assert N.x.angle_between(A.x + A.y) == 0
    assert (A.x + A.y).express(N) == sqrt(2) * N.x
    assert A.dcm(N) == Matrix([[sqrt(2) / 2, -sqrt(2) / 2, 0], [sqrt(2) / 2, sqrt(2) / 2, 0], [0, 0, 1]])
    assert C.masscenter.pos_from(P.masscenter) == (q + 1) * N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N) == (q - sqrt(2) / 2 + 1) * N.x + sqrt(2) / 2 * N.y
    assert C.masscenter.vel(N).express(A) == u * (A.x + A.y) / sqrt(2)
    assert C.masscenter.vel(N) == u * N.x
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0
    N, A, P, C = _generate_body()
    PrismaticJoint('S', P, C, parent_point=N.x, child_point=A.x, child_interframe=A.x + A.y - A.z)
    assert N.x.angle_between(A.x + A.y - A.z) == 0
    assert (A.x + A.y - A.z).express(N) == sqrt(3) * N.x
    assert _simplify_matrix(A.dcm(N)) == Matrix([[sqrt(3) / 3, -sqrt(3) / 3, sqrt(3) / 3], [sqrt(3) / 3, sqrt(3) / 6 + S(1) / 2, S(1) / 2 - sqrt(3) / 6], [-sqrt(3) / 3, S(1) / 2 - sqrt(3) / 6, sqrt(3) / 6 + S(1) / 2]])
    assert C.masscenter.pos_from(P.masscenter) == (q + 1) * N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N) == (q - sqrt(3) / 3 + 1) * N.x + sqrt(3) / 3 * N.y - sqrt(3) / 3 * N.z
    assert C.masscenter.vel(N) == u * N.x
    assert C.masscenter.vel(N).express(A) == sqrt(3) * u / 3 * A.x + sqrt(3) * u / 3 * A.y - sqrt(3) * u / 3 * A.z
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0
    N, A, P, C = _generate_body()
    m, n = symbols('m n')
    PrismaticJoint('S', P, C, parent_point=m * N.x, child_point=n * A.x, child_interframe=A.x + A.y - A.z, parent_interframe=N.x - N.y + N.z)
    assert (N.x - N.y + N.z).angle_between(A.x + A.y - A.z).simplify() == 0
    assert (A.x + A.y - A.z).express(N) == N.x - N.y + N.z
    assert _simplify_matrix(A.dcm(N)) == Matrix([[-S(1) / 3, -S(2) / 3, S(2) / 3], [S(2) / 3, S(1) / 3, S(2) / 3], [-S(2) / 3, S(2) / 3, S(1) / 3]])
    assert C.masscenter.pos_from(P.masscenter) == (m + sqrt(3) * q / 3) * N.x - sqrt(3) * q / 3 * N.y + sqrt(3) * q / 3 * N.z - n * A.x
    assert C.masscenter.pos_from(P.masscenter).express(N) == (m + n / 3 + sqrt(3) * q / 3) * N.x + (2 * n / 3 - sqrt(3) * q / 3) * N.y + (-2 * n / 3 + sqrt(3) * q / 3) * N.z
    assert C.masscenter.vel(N) == sqrt(3) * u / 3 * N.x - sqrt(3) * u / 3 * N.y + sqrt(3) * u / 3 * N.z
    assert C.masscenter.vel(N).express(A) == sqrt(3) * u / 3 * A.x + sqrt(3) * u / 3 * A.y - sqrt(3) * u / 3 * A.z
    assert A.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == 0