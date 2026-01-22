from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.matrices import ImmutableMatrix, Matrix
from sympy.physics.control import (TransferFunction, Series, Parallel,
from sympy.testing.pytest import raises
def test_Parallel_functions():
    tf1 = TransferFunction(1, s ** 2 + 2 * zeta * wn * s + wn ** 2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2 * p - s, a2 * s + p, s)
    tf4 = TransferFunction(a0 * p + p ** a1 - s, p, p)
    tf5 = TransferFunction(a1 * s ** 2 + a2 * s - a0, s + a0, s)
    assert tf1 + tf2 + tf3 == Parallel(tf1, tf2, tf3)
    assert tf1 + tf2 + tf3 + tf5 == Parallel(tf1, tf2, tf3, tf5)
    assert tf1 + tf2 - tf3 - tf5 == Parallel(tf1, tf2, -tf3, -tf5)
    assert tf1 + tf2 * tf3 == Parallel(tf1, Series(tf2, tf3))
    assert tf1 - tf2 * tf3 == Parallel(tf1, -Series(tf2, tf3))
    assert -tf1 - tf2 == Parallel(-tf1, -tf2)
    assert -(tf1 + tf2) == Series(TransferFunction(-1, 1, s), Parallel(tf1, tf2))
    assert (tf2 + tf3) * tf1 == Series(Parallel(tf2, tf3), tf1)
    assert (tf1 + tf2) * (tf3 * tf5) == Series(Parallel(tf1, tf2), tf3, tf5)
    assert -(tf2 + tf3) * -tf5 == Series(TransferFunction(-1, 1, s), Parallel(tf2, tf3), -tf5)
    assert tf2 + tf3 + tf2 * tf1 + tf5 == Parallel(tf2, tf3, Series(tf2, tf1), tf5)
    assert tf2 + tf3 + tf2 * tf1 - tf3 == Parallel(tf2, tf3, Series(tf2, tf1), -tf3)
    assert (tf1 + tf2 + tf5) * (tf3 + tf5) == Series(Parallel(tf1, tf2, tf5), Parallel(tf3, tf5))
    raises(ValueError, lambda: tf1 + tf2 + tf4)
    raises(ValueError, lambda: tf1 - tf2 * tf4)
    raises(ValueError, lambda: tf3 + Matrix([1, 2, 3]))
    assert Parallel(tf1, tf2, evaluate=True) == Parallel(tf1, tf2).doit() == TransferFunction(k * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + 1, s ** 2 + 2 * s * wn * zeta + wn ** 2, s)
    assert Parallel(tf1, tf2, Series(-tf1, tf3), evaluate=True) == Parallel(tf1, tf2, Series(-tf1, tf3)).doit() == TransferFunction(k * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2 + (-a2 * p + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2, s)
    assert Parallel(tf2, tf1, -tf3, evaluate=True) == Parallel(tf2, tf1, -tf3).doit() == TransferFunction(a2 * s + k * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + p + (-a2 * p + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert not Parallel(tf1, -tf2, evaluate=False) == Parallel(tf1, -tf2).doit()
    assert Parallel(Series(tf1, tf2), Series(tf2, tf3)).doit() == TransferFunction(k * (a2 * p - s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + k * (a2 * s + p), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Parallel(-tf1, -tf2, -tf3).doit() == TransferFunction(-a2 * s - k * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) - p + (-a2 * p + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert -Parallel(tf1, tf2, tf3).doit() == TransferFunction(-a2 * s - k * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) - p - (a2 * p - s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Parallel(tf2, tf3, Series(tf2, -tf1), tf3).doit() == TransferFunction(k * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) - k * (a2 * s + p) + (2 * a2 * p - 2 * s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Parallel(tf1, tf2).rewrite(TransferFunction) == TransferFunction(k * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + 1, s ** 2 + 2 * s * wn * zeta + wn ** 2, s)
    assert Parallel(tf2, tf1, -tf3).rewrite(TransferFunction) == TransferFunction(a2 * s + k * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + p + (-a2 * p + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Parallel(tf1, Parallel(tf2, tf3)) == Parallel(tf1, tf2, tf3) == Parallel(Parallel(tf1, tf2), tf3)
    P1 = Parallel(Series(tf1, tf2), Series(tf2, tf3))
    assert P1.is_proper
    assert not P1.is_strictly_proper
    assert P1.is_biproper
    P2 = Parallel(tf1, -tf2, -tf3)
    assert P2.is_proper
    assert not P2.is_strictly_proper
    assert P2.is_biproper
    P3 = Parallel(tf1, -tf2, Series(tf1, tf3))
    assert P3.is_proper
    assert not P3.is_strictly_proper
    assert P3.is_biproper