from math import prod
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Dagger, Commutator, qapply
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.boson import (
def test_boson_states():
    a = BosonOp('a')
    n = 3
    assert (BosonFockBra(0) * BosonFockKet(1)).doit() == 0
    assert (BosonFockBra(1) * BosonFockKet(1)).doit() == 1
    assert qapply(BosonFockBra(n) * Dagger(a) ** n * BosonFockKet(0)) == sqrt(prod(range(1, n + 1)))
    alpha1, alpha2 = (1.2, 4.3)
    assert (BosonCoherentBra(alpha1) * BosonCoherentKet(alpha1)).doit() == 1
    assert (BosonCoherentBra(alpha2) * BosonCoherentKet(alpha2)).doit() == 1
    assert abs((BosonCoherentBra(alpha1) * BosonCoherentKet(alpha2)).doit() - exp((alpha1 - alpha2) ** 2 * Rational(-1, 2))) < 1e-12
    assert qapply(a * BosonCoherentKet(alpha1)) == alpha1 * BosonCoherentKet(alpha1)