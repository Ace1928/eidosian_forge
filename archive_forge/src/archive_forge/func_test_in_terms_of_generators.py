from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_in_terms_of_generators():
    R = QQ.old_poly_ring(x, order='ilex')
    M = R.free_module(2).submodule([2 * x, 0], [1, 2])
    assert M.in_terms_of_generators([x, x]) == [R.convert(Rational(1, 4)), R.convert(x / 2)]
    raises(ValueError, lambda: M.in_terms_of_generators([1, 0]))
    M = R.free_module(2) / ([x, 0], [1, 1])
    SM = M.submodule([1, x])
    assert SM.in_terms_of_generators([2, 0]) == [R.convert(-2 / (x - 1))]
    R = QQ.old_poly_ring(x, y) / [x ** 2 - y ** 2]
    M = R.free_module(2)
    SM = M.submodule([x, 0], [0, y])
    assert SM.in_terms_of_generators([x ** 2, x ** 2]) == [R.convert(x), R.convert(y)]