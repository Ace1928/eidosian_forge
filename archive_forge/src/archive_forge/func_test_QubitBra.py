import random
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qubit import (measure_all, measure_partial,
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.shor import Qubit
from sympy.testing.pytest import raises
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
def test_QubitBra():
    qb = Qubit(0)
    qb_bra = QubitBra(0)
    assert qb.dual_class() == QubitBra
    assert qb_bra.dual_class() == Qubit
    qb = Qubit(1, 1, 0)
    qb_bra = QubitBra(1, 1, 0)
    assert represent(qb, nqubits=3).H == represent(qb_bra, nqubits=3)
    qb = Qubit(0, 1)
    qb_bra = QubitBra(1, 0)
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(0)
    qb_bra = QubitBra(0, 1)
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(1)