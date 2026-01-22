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
def test_pin_joint_chaos_pendulum():
    mA, mB, lA, lB, h = symbols('mA, mB, lA, lB, h')
    theta, phi, omega, alpha = dynamicsymbols('theta phi omega alpha')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    lA = (lB - h / 2) / 2
    lC = lB / 2 + h / 4
    rod = Body('rod', frame=A, mass=mA)
    plate = Body('plate', mass=mB, frame=B)
    C = Body('C', frame=N)
    J1 = PinJoint('J1', C, rod, coordinates=theta, speeds=omega, child_point=lA * A.z, joint_axis=N.y)
    J2 = PinJoint('J2', rod, plate, coordinates=phi, speeds=alpha, parent_point=lC * A.z, joint_axis=A.z)
    assert A.dcm(N) == Matrix([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
    assert A.dcm(B) == Matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    assert B.dcm(N) == Matrix([[cos(phi) * cos(theta), sin(phi), -sin(theta) * cos(phi)], [-sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta)], [sin(theta), 0, cos(theta)]])
    assert A.ang_vel_in(N) == omega * N.y
    assert A.ang_vel_in(B) == -alpha * A.z
    assert N.ang_vel_in(B) == -omega * N.y - alpha * A.z
    assert J1.kdes == Matrix([omega - theta.diff(t)])
    assert J2.kdes == Matrix([alpha - phi.diff(t)])
    assert C.masscenter.pos_from(rod.masscenter) == lA * A.z
    assert rod.masscenter.pos_from(plate.masscenter) == -lC * A.z
    assert rod.masscenter.vel(N) == (h / 4 - lB / 2) * omega * A.x
    assert plate.masscenter.vel(N) == ((h / 4 - lB / 2) * omega + (h / 4 + lB / 2) * omega) * A.x