from sympy.polys import QQ, ilex
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_nontriv_local():
    R = QQ.old_poly_ring(x, y, z, order=ilex)

    def contains(I, f):
        return R.ideal(*I).contains(f)
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x ** 2 + y, x ** 2 + x], x - y)
    assert not contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 2)
    assert contains([x * (1 + x + y), y * (1 + z)], x)
    assert contains([x * (1 + x + y), y * (1 + z)], x + y)