from sympy.vector import CoordSys3D, Gradient, Divergence, Curl, VectorZero, Laplacian
from sympy.printing.repr import srepr
def test_Gradient():
    assert Gradient(s1) == Gradient(R.x * R.y * R.z)
    assert Gradient(s2) == Gradient(R.x + 3 * R.y ** 2)
    assert Gradient(s1).doit() == R.y * R.z * R.i + R.x * R.z * R.j + R.x * R.y * R.k
    assert Gradient(s2).doit() == R.i + 6 * R.y * R.j