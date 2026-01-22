from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_eq_hash():
    p1 = Poly(x ** 2 - 2, x, domain=ZZ)
    p2 = Poly(x ** 2 - 2, x, domain=QQ)
    K1 = FiniteExtension(p1)
    K2 = FiniteExtension(p2)
    assert K1 == FiniteExtension(Poly(x ** 2 - 2))
    assert K2 != FiniteExtension(Poly(x ** 2 - 2))
    assert len({K1, K2, FiniteExtension(p1)}) == 2