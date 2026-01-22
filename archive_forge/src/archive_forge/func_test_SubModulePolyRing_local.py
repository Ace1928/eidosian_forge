from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_SubModulePolyRing_local():
    R = QQ.old_poly_ring(x, y, order=ilex)
    F = R.free_module(3)
    Fd = F.submodule([1 + x, 0, 0], [1 + y, 2 + 2 * y, 0], [1, 2, 3])
    M = F.submodule([x ** 2 + y ** 2, 1, 0], [x, y, 1])
    assert F == Fd
    assert Fd == F
    assert F != M
    assert M != F
    assert Fd != M
    assert M != Fd
    assert Fd == F.submodule(*F.basis())
    assert Fd.is_full_module()
    assert not M.is_full_module()
    assert not Fd.is_zero()
    assert not M.is_zero()
    assert Fd.submodule().is_zero()
    assert M.contains([x ** 2 + y ** 2 + x, 1 + y, 1])
    assert not M.contains([x ** 2 + y ** 2 + x, 1 + y, 2])
    assert M.contains([y ** 2, 1 - x * y, -x])
    assert F.submodule([1 + x, 0, 0]) == F.submodule([1, 0, 0])
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1 + x * y])) == F
    raises(ValueError, lambda: M.submodule([1, 0, 0]))