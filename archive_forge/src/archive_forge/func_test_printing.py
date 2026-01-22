from sympy.core.singleton import S
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
from sympy.polys.agca import homomorphism
from sympy.testing.pytest import raises
def test_printing():
    R = QQ.old_poly_ring(x)
    assert str(homomorphism(R.free_module(1), R.free_module(1), [0])) == 'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1'
    assert str(homomorphism(R.free_module(2), R.free_module(2), [0, 0])) == 'Matrix([                       \n[0, 0], : QQ[x]**2 -> QQ[x]**2\n[0, 0]])                       '
    assert str(homomorphism(R.free_module(1), R.free_module(1) / [[x]], [0])) == 'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1/<[x]>'
    assert str(R.free_module(0).identity_hom()) == 'Matrix(0, 0, []) : QQ[x]**0 -> QQ[x]**0'