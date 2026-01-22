from sympy.core.singleton import S
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
from sympy.polys.agca import homomorphism
from sympy.testing.pytest import raises
def test_operations():
    F = QQ.old_poly_ring(x).free_module(2)
    G = QQ.old_poly_ring(x).free_module(3)
    f = F.identity_hom()
    g = homomorphism(F, F, [0, [1, x]])
    h = homomorphism(F, F, [[1, 0], 0])
    i = homomorphism(F, G, [[1, 0, 0], [0, 1, 0]])
    assert f == f
    assert f != g
    assert f != i
    assert (f != F.identity_hom()) is False
    assert 2 * f == f * 2 == homomorphism(F, F, [[2, 0], [0, 2]])
    assert f / 2 == homomorphism(F, F, [[S.Half, 0], [0, S.Half]])
    assert f + g == homomorphism(F, F, [[1, 0], [1, x + 1]])
    assert f - g == homomorphism(F, F, [[1, 0], [-1, 1 - x]])
    assert f * g == g == g * f
    assert h * g == homomorphism(F, F, [0, [1, 0]])
    assert g * h == homomorphism(F, F, [0, 0])
    assert i * f == i
    assert f([1, 2]) == [1, 2]
    assert g([1, 2]) == [2, 2 * x]
    assert i.restrict_domain(F.submodule([x, x]))([x, x]) == i([x, x])
    h1 = h.quotient_domain(F.submodule([0, 1]))
    assert h1([1, 0]) == h([1, 0])
    assert h1.restrict_domain(h1.domain.submodule([x, 0]))([x, 0]) == h([x, 0])
    raises(TypeError, lambda: f / g)
    raises(TypeError, lambda: f + 1)
    raises(TypeError, lambda: f + i)
    raises(TypeError, lambda: f - 1)
    raises(TypeError, lambda: f * i)