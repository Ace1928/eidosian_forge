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
def test_TransferFunctionMatrix_functions():
    tf5 = TransferFunction(a1 * s ** 2 + a2 * s - a0, s + a0, s)
    mat_1 = ImmutableMatrix([[s * (s + 1) * (s - 3) / (s ** 4 + 1), 2], [p, p * (s + 1) / (s * (s ** 1 + 1))]])
    mat_2 = ImmutableMatrix([[(2 * s + 1) / (s ** 2 - 9)]])
    mat_3 = ImmutableMatrix([[1, 2], [3, 4]])
    assert TransferFunctionMatrix.from_Matrix(mat_1, s) == TransferFunctionMatrix([[TransferFunction(s * (s - 3) * (s + 1), s ** 4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(p, 1, s), TransferFunction(p, s, s)]])
    assert TransferFunctionMatrix.from_Matrix(mat_2, s) == TransferFunctionMatrix([[TransferFunction(2 * s + 1, s ** 2 - 9, s)]])
    assert TransferFunctionMatrix.from_Matrix(mat_3, p) == TransferFunctionMatrix([[TransferFunction(1, 1, p), TransferFunction(2, 1, p)], [TransferFunction(3, 1, p), TransferFunction(4, 1, p)]])
    tfm1 = TransferFunctionMatrix([[TF1], [TF2]])
    assert -tfm1 == TransferFunctionMatrix([[-TF1], [-TF2]])
    tfm2 = TransferFunctionMatrix([[TF1, TF2, TF3], [tf5, -TF1, -TF3]])
    assert -tfm2 == TransferFunctionMatrix([[-TF1, -TF2, -TF3], [-tf5, TF1, TF3]])
    H_1 = TransferFunctionMatrix.from_Matrix(mat_1, s)
    H_2 = TransferFunctionMatrix([[TransferFunction(a * p * s, k * s ** 2, s), TransferFunction(p * s, k * (s ** 2 - a), s)]])
    assert H_1.subs(p, 1) == TransferFunctionMatrix([[TransferFunction(s * (s - 3) * (s + 1), s ** 4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])
    assert H_1.subs({p: 1}) == TransferFunctionMatrix([[TransferFunction(s * (s - 3) * (s + 1), s ** 4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])
    assert H_1.subs({p: 1, s: 1}) == TransferFunctionMatrix([[TransferFunction(s * (s - 3) * (s + 1), s ** 4 + 1, s), TransferFunction(2, 1, s)], [TransferFunction(1, 1, s), TransferFunction(1, s, s)]])
    assert H_2.subs(p, 2) == TransferFunctionMatrix([[TransferFunction(2 * a * s, k * s ** 2, s), TransferFunction(2 * s, k * (-a + s ** 2), s)]])
    assert H_2.subs(k, 1) == TransferFunctionMatrix([[TransferFunction(a * p * s, s ** 2, s), TransferFunction(p * s, -a + s ** 2, s)]])
    assert H_2.subs(a, 0) == TransferFunctionMatrix([[TransferFunction(0, k * s ** 2, s), TransferFunction(p * s, k * s ** 2, s)]])
    assert H_2.subs({p: 1, k: 1, a: a0}) == TransferFunctionMatrix([[TransferFunction(a0 * s, s ** 2, s), TransferFunction(s, -a0 + s ** 2, s)]])
    assert H_1.transpose() == TransferFunctionMatrix([[TransferFunction(s * (s - 3) * (s + 1), s ** 4 + 1, s), TransferFunction(p, 1, s)], [TransferFunction(2, 1, s), TransferFunction(p, s, s)]])
    assert H_2.transpose() == TransferFunctionMatrix([[TransferFunction(a * p * s, k * s ** 2, s)], [TransferFunction(p * s, k * (-a + s ** 2), s)]])
    assert H_1.transpose().transpose() == H_1
    assert H_2.transpose().transpose() == H_2
    assert H_1.elem_poles() == [[[-sqrt(2) / 2 - sqrt(2) * I / 2, -sqrt(2) / 2 + sqrt(2) * I / 2, sqrt(2) / 2 - sqrt(2) * I / 2, sqrt(2) / 2 + sqrt(2) * I / 2], []], [[], [0]]]
    assert H_2.elem_poles() == [[[0, 0], [sqrt(a), -sqrt(a)]]]
    assert tfm2.elem_poles() == [[[wn * (-zeta + sqrt((zeta - 1) * (zeta + 1))), wn * (-zeta - sqrt((zeta - 1) * (zeta + 1)))], [], [-p / a2]], [[-a0], [wn * (-zeta + sqrt((zeta - 1) * (zeta + 1))), wn * (-zeta - sqrt((zeta - 1) * (zeta + 1)))], [-p / a2]]]
    assert H_1.elem_zeros() == [[[-1, 0, 3], []], [[], []]]
    assert H_2.elem_zeros() == [[[0], [0]]]
    assert tfm2.elem_zeros() == [[[], [], [a2 * p]], [[-a2 / (2 * a1) - sqrt(4 * a0 * a1 + a2 ** 2) / (2 * a1), -a2 / (2 * a1) + sqrt(4 * a0 * a1 + a2 ** 2) / (2 * a1)], [], [a2 * p]]]
    H_3 = TransferFunctionMatrix([[Series(TransferFunction(1, s ** 3 - 3, s), TransferFunction(s ** 2 - 2 * s + 5, 1, s), TransferFunction(1, s, s))]])
    H_4 = TransferFunctionMatrix([[Parallel(TransferFunction(s ** 3 - 3, 4 * s ** 4 - s ** 2 - 2 * s + 5, s), TransferFunction(4 - s ** 3, 4 * s ** 4 - s ** 2 - 2 * s + 5, s))]])
    assert H_3.doit() == TransferFunctionMatrix([[TransferFunction(s ** 2 - 2 * s + 5, s * (s ** 3 - 3), s)]])
    assert H_4.doit() == TransferFunctionMatrix([[TransferFunction(1, 4 * s ** 4 - s ** 2 - 2 * s + 5, s)]])
    assert H_1._flat() == [TransferFunction(s * (s - 3) * (s + 1), s ** 4 + 1, s), TransferFunction(2, 1, s), TransferFunction(p, 1, s), TransferFunction(p, s, s)]
    assert H_2._flat() == [TransferFunction(a * p * s, k * s ** 2, s), TransferFunction(p * s, k * (-a + s ** 2), s)]
    assert H_3._flat() == [Series(TransferFunction(1, s ** 3 - 3, s), TransferFunction(s ** 2 - 2 * s + 5, 1, s), TransferFunction(1, s, s))]
    assert H_4._flat() == [Parallel(TransferFunction(s ** 3 - 3, 4 * s ** 4 - s ** 2 - 2 * s + 5, s), TransferFunction(4 - s ** 3, 4 * s ** 4 - s ** 2 - 2 * s + 5, s))]
    assert H_1.evalf() == TransferFunctionMatrix(((TransferFunction(s * (s - 3.0) * (s + 1.0), s ** 4 + 1.0, s), TransferFunction(2.0, 1, s)), (TransferFunction(1.0 * p, 1, s), TransferFunction(p, s, s))))
    assert H_2.subs({a: 3.141, p: 2.88, k: 2}).evalf() == TransferFunctionMatrix(((TransferFunction(4.52304, s, s), TransferFunction(2.88 * s, 2.0 * s ** 2 - 6.282, s)),))
    H_5 = TransferFunctionMatrix([[TransferFunction(s ** 5 + s ** 3 + s, s - s ** 2, s), TransferFunction((s + 3) * (s - 1), (s - 1) * (s + 5), s)]])
    assert H_5.simplify() == simplify(H_5) == TransferFunctionMatrix(((TransferFunction(-s ** 4 - s ** 2 - 1, s - 1, s), TransferFunction(s + 3, s + 5, s)),))
    assert H_1.expand() == TransferFunctionMatrix(((TransferFunction(s ** 3 - 2 * s ** 2 - 3 * s, s ** 4 + 1, s), TransferFunction(2, 1, s)), (TransferFunction(p, 1, s), TransferFunction(p, s, s))))
    assert H_5.expand() == TransferFunctionMatrix(((TransferFunction(s ** 5 + s ** 3 + s, -s ** 2 + s, s), TransferFunction(s ** 2 + 2 * s - 3, s ** 2 + 4 * s - 5, s)),))