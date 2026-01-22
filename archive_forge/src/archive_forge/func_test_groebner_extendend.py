from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_groebner_extendend():
    M = QQ.old_poly_ring(x, y, z).free_module(3).submodule([x + 1, y, 1], [x * y, z, z ** 2])
    G, R = M._groebner_vec(extended=True)
    for i, g in enumerate(G):
        assert g == sum((c * gen for c, gen in zip(R[i], M.gens)))