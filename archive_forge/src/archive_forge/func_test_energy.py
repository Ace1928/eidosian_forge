from sympy.physics.pring import wavefunction, energy
from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.abc import m, x, r
from sympy.physics.quantum.constants import hbar
def test_energy(n=1):
    for i in range(n + 1):
        assert simplify(energy(i, m, r) - i ** 2 * hbar ** 2 / (2 * m * r ** 2)) == 0