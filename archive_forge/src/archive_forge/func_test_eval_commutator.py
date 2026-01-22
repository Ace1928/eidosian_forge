from sympy.core.numbers import Integer
from sympy.core.symbol import symbols
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.anticommutator import AntiCommutator as AComm
from sympy.physics.quantum.operator import Operator
def test_eval_commutator():
    F = Foo('F')
    B = Bar('B')
    T = Tam('T')
    assert AComm(F, B).doit() == 0
    assert AComm(B, F).doit() == 0
    assert AComm(F, T).doit() == 1
    assert AComm(T, F).doit() == 1
    assert AComm(B, T).doit() == B * T + T * B