from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.physics.vector import ReferenceFrame, Vector, Point, \
from sympy.physics.vector.fieldfunctions import divergence, \
from sympy.testing.pytest import raises
def test_curl():
    assert curl(Vector(0), R) == Vector(0)
    assert curl(R.x, R) == Vector(0)
    assert curl(2 * R[1] ** 2 * R.y, R) == Vector(0)
    assert curl(R[0] * R[1] * R.z, R) == R[0] * R.x - R[1] * R.y
    assert curl(R[0] * R[1] * R[2] * (R.x + R.y + R.z), R) == (-R[0] * R[1] + R[0] * R[2]) * R.x + (R[0] * R[1] - R[1] * R[2]) * R.y + (-R[0] * R[2] + R[1] * R[2]) * R.z
    assert curl(2 * R[0] ** 2 * R.y, R) == 4 * R[0] * R.z
    assert curl(P[0] ** 2 * R.x + P.y, R) == -2 * (R[0] * cos(q) + R[1] * sin(q)) * sin(q) * R.z
    assert curl(P[0] * R.y, P) == cos(q) * P.z