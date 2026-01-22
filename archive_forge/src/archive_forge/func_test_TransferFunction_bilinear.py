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
def test_TransferFunction_bilinear():
    tf = TransferFunction(1, a * s + b, s)
    numZ, denZ = bilinear(tf, T)
    tf_test_bilinear = TransferFunction(s * numZ[0] + numZ[1], s * denZ[0] + denZ[1], s)
    tf_test_manual = TransferFunction(s * T + T, s * (T * b + 2 * a) + T * b - 2 * a, s)
    assert S.Zero == (tf_test_bilinear - tf_test_manual).simplify().num