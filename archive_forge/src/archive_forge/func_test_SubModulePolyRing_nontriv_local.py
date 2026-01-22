from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_SubModulePolyRing_nontriv_local():
    R = QQ.old_poly_ring(x, y, z, order=ilex)
    F = R.free_module(1)

    def contains(I, f):
        return F.submodule(*[[g] for g in I]).contains([f])
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x ** 2 + y, x ** 2 + x], x - y)
    assert not contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 2)
    assert contains([x * (1 + x + y), y * (1 + z)], x)
    assert contains([x * (1 + x + y), y * (1 + z)], x + y)