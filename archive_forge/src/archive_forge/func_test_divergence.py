from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.physics.vector import ReferenceFrame, Vector, Point, \
from sympy.physics.vector.fieldfunctions import divergence, \
from sympy.testing.pytest import raises
def test_divergence():
    assert divergence(Vector(0), R) is S.Zero
    assert divergence(R.x, R) is S.Zero
    assert divergence(R[0] ** 2 * R.x, R) == 2 * R[0]
    assert divergence(R[0] * R[1] * R[2] * (R.x + R.y + R.z), R) == R[0] * R[1] + R[0] * R[2] + R[1] * R[2]
    assert divergence(1 / (R[0] * R[1] * R[2]) * (R.x + R.y + R.z), R) == -1 / (R[0] * R[1] * R[2] ** 2) - 1 / (R[0] * R[1] ** 2 * R[2]) - 1 / (R[0] ** 2 * R[1] * R[2])
    v = P[0] * P.x + P[1] * P.y + P[2] * P.z
    assert divergence(v, P) == 3
    assert divergence(v, R).simplify() == 3
    assert divergence(P[0] * R.x + R[0] * P.x, R) == 2 * cos(q)