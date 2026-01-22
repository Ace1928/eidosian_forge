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
def test_Feedback_functions():
    tf = TransferFunction(1, 1, s)
    tf1 = TransferFunction(1, s ** 2 + 2 * zeta * wn * s + wn ** 2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2 * p - s, a2 * s + p, s)
    tf4 = TransferFunction(a0 * p + p ** a1 - s, p, p)
    tf5 = TransferFunction(a1 * s ** 2 + a2 * s - a0, s + a0, s)
    tf6 = TransferFunction(s - p, p + s, p)
    assert tf / (tf + tf1) == Feedback(tf, tf1)
    assert tf / (tf + tf1 * tf2 * tf3) == Feedback(tf, tf1 * tf2 * tf3)
    assert tf1 / (tf + tf1 * tf2 * tf3) == Feedback(tf1, tf2 * tf3)
    assert tf1 * tf2 / (tf + tf1 * tf2) == Feedback(tf1 * tf2, tf)
    assert tf1 * tf2 / (tf + tf1 * tf2 * tf5) == Feedback(tf1 * tf2, tf5)
    assert tf1 * tf2 / (tf + tf1 * tf2 * tf5 * tf3) in (Feedback(tf1 * tf2, tf5 * tf3), Feedback(tf1 * tf2, tf3 * tf5))
    assert tf4 / (TransferFunction(1, 1, p) + tf4 * tf6) == Feedback(tf4, tf6)
    assert tf5 / (tf + tf5) == Feedback(tf5, tf)
    raises(TypeError, lambda: tf1 * tf2 * tf3 / (1 + tf1 * tf2 * tf3))
    raises(ValueError, lambda: tf1 * tf2 * tf3 / tf3 * tf5)
    raises(ValueError, lambda: tf2 * tf3 / (tf + tf2 * tf3 * tf4))
    assert Feedback(tf, tf1 * tf2 * tf3).doit() == TransferFunction((a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), k * (a2 * p - s) + (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Feedback(tf, tf1 * tf2 * tf3).sensitivity == 1 / (k * (a2 * p - s) / ((a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)) + 1)
    assert Feedback(tf1, tf2 * tf3).doit() == TransferFunction((a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (k * (a2 * p - s) + (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Feedback(tf1, tf2 * tf3).sensitivity == 1 / (k * (a2 * p - s) / ((a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)) + 1)
    assert Feedback(tf1 * tf2, tf5).doit() == TransferFunction(k * (a0 + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (k * (-a0 + a1 * s ** 2 + a2 * s) + (a0 + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Feedback(tf1 * tf2, tf5, 1).sensitivity == 1 / (-k * (-a0 + a1 * s ** 2 + a2 * s) / ((a0 + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)) + 1)
    assert Feedback(tf4, tf6).doit() == TransferFunction(p * (p + s) * (a0 * p + p ** a1 - s), p * (p * (p + s) + (-p + s) * (a0 * p + p ** a1 - s)), p)
    assert -Feedback(tf4 * tf6, TransferFunction(1, 1, p)).doit() == TransferFunction(-p * (-p + s) * (p + s) * (a0 * p + p ** a1 - s), p * (p + s) * (p * (p + s) + (-p + s) * (a0 * p + p ** a1 - s)), p)
    assert Feedback(tf, tf).doit() == TransferFunction(1, 2, s)
    assert Feedback(tf1, tf2 * tf5).rewrite(TransferFunction) == TransferFunction((a0 + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (k * (-a0 + a1 * s ** 2 + a2 * s) + (a0 + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Feedback(TransferFunction(1, 1, p), tf4).rewrite(TransferFunction) == TransferFunction(p, a0 * p + p + p ** a1 - s, p)