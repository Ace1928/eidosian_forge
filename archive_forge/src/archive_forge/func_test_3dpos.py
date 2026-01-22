from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval
from sympy.physics.quantum import qapply, represent, L2, Dagger
from sympy.physics.quantum import Commutator, hbar
from sympy.physics.quantum.cartesian import (
from sympy.physics.quantum.operator import DifferentialOperator
def test_3dpos():
    assert Y.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert Z.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    test_ket = PositionKet3D(x, y, z)
    assert qapply(X * test_ket) == x * test_ket
    assert qapply(Y * test_ket) == y * test_ket
    assert qapply(Z * test_ket) == z * test_ket
    assert qapply(X * Y * test_ket) == x * y * test_ket
    assert qapply(X * Y * Z * test_ket) == x * y * z * test_ket
    assert qapply(Y * Z * test_ket) == y * z * test_ket
    assert PositionKet3D() == test_ket
    assert YOp() == Y
    assert ZOp() == Z
    assert PositionKet3D.dual_class() == PositionBra3D
    assert PositionBra3D.dual_class() == PositionKet3D
    other_ket = PositionKet3D(x_1, y_1, z_1)
    assert (Dagger(other_ket) * test_ket).doit() == DiracDelta(x - x_1) * DiracDelta(y - y_1) * DiracDelta(z - z_1)
    assert test_ket.position_x == x
    assert test_ket.position_y == y
    assert test_ket.position_z == z
    assert other_ket.position_x == x_1
    assert other_ket.position_y == y_1
    assert other_ket.position_z == z_1