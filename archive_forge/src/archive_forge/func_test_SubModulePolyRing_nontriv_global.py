from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_SubModulePolyRing_nontriv_global():
    R = QQ.old_poly_ring(x, y, z)
    F = R.free_module(1)

    def contains(I, f):
        return F.submodule(*[[g] for g in I]).contains([f])
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x ** 2 + y, x ** 2 + x], x - y)
    assert not contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 2)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 3)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 4)
    assert not contains([x + y + z, x * y + x * z + y * z, x * y * z], x * y ** 2)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 4 + y ** 3 + 2 * z * y * x)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x * y * z)
    assert contains([x, 1 + x + y, 5 - 7 * y], 1)
    assert contains([x ** 3 + y ** 3, y ** 3 + z ** 3, z ** 3 + x ** 3, x ** 2 * y + x ** 2 * z + y ** 2 * z], x ** 3)
    assert not contains([x ** 3 + y ** 3, y ** 3 + z ** 3, z ** 3 + x ** 3, x ** 2 * y + x ** 2 * z + y ** 2 * z], x ** 2 + y ** 2)
    assert not contains([x * (1 + x + y), y * (1 + z)], x)
    assert not contains([x * (1 + x + y), y * (1 + z)], x + y)