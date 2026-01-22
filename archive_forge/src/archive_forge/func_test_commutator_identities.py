from sympy.core.numbers import Integer
from sympy.core.symbol import symbols
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.anticommutator import AntiCommutator as AComm
from sympy.physics.quantum.operator import Operator
def test_commutator_identities():
    assert AComm(a * A, b * B) == a * b * AComm(A, B)
    assert AComm(A, A) == 2 * A ** 2
    assert AComm(A, B) == AComm(B, A)
    assert AComm(a, b) == 2 * a * b
    assert AComm(A, B).doit() == A * B + B * A