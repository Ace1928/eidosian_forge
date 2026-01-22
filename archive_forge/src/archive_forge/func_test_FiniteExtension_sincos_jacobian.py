from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_sincos_jacobian():
    r, p, t = symbols('rho, phi, theta')
    elements = [[sin(p) * cos(t), r * cos(p) * cos(t), -r * sin(p) * sin(t)], [sin(p) * sin(t), r * cos(p) * sin(t), r * sin(p) * cos(t)], [cos(p), -r * sin(p), 0]]

    def make_extension(K):
        K = FiniteExtension(Poly(sin(p) ** 2 + cos(p) ** 2 - 1, sin(p), domain=K[cos(p)]))
        K = FiniteExtension(Poly(sin(t) ** 2 + cos(t) ** 2 - 1, sin(t), domain=K[cos(t)]))
        return K
    Ksc1 = make_extension(ZZ[r])
    Ksc2 = make_extension(ZZ)[r]
    for K in [Ksc1, Ksc2]:
        elements_K = [[K.convert(e) for e in row] for row in elements]
        J = DomainMatrix(elements_K, (3, 3), K)
        det = J.charpoly()[-1] * (-K.one) ** 3
        assert det == K.convert(r ** 2 * sin(p))