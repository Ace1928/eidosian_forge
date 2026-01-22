from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.abc import omega, m, x
from sympy.physics.qho_1d import psi_n, E_n, coherent_state
from sympy.physics.quantum.constants import hbar
def test_orthogonality(n=1):
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            assert integrate(psi_n(i, x, 1, 1) * psi_n(j, x, 1, 1), (x, -oo, oo)) == 0