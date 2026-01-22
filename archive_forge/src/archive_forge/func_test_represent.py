from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises
def test_represent():
    assert represent(sx) == Matrix([[0, 1], [1, 0]])
    assert represent(sy) == Matrix([[0, -I], [I, 0]])
    assert represent(sz) == Matrix([[1, 0], [0, -1]])
    assert represent(sm) == Matrix([[0, 0], [1, 0]])
    assert represent(sp) == Matrix([[0, 1], [0, 0]])