from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.physics.hydrogen import R_nl, E_nl, E_nl_dirac, Psi_nlm
from sympy.testing.pytest import raises
def test_hydrogen_energies():
    assert E_nl(n, Z) == -Z ** 2 / (2 * n ** 2)
    assert E_nl(n) == -1 / (2 * n ** 2)
    assert E_nl(1, 47) == -S(47) ** 2 / (2 * 1 ** 2)
    assert E_nl(2, 47) == -S(47) ** 2 / (2 * 2 ** 2)
    assert E_nl(1) == -S.One / (2 * 1 ** 2)
    assert E_nl(2) == -S.One / (2 * 2 ** 2)
    assert E_nl(3) == -S.One / (2 * 3 ** 2)
    assert E_nl(4) == -S.One / (2 * 4 ** 2)
    assert E_nl(100) == -S.One / (2 * 100 ** 2)
    raises(ValueError, lambda: E_nl(0))