from sympy.core.singleton import S
from sympy.physics.quantum.operatorset import (
from sympy.physics.quantum.cartesian import (
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.spin import (
from sympy.testing.pytest import raises
def test_op_to_state():
    assert operators_to_state(XOp) == XKet()
    assert operators_to_state(PxOp) == PxKet()
    assert operators_to_state(Operator) == Ket()
    assert state_to_operators(operators_to_state(XOp('Q'))) == XOp('Q')
    assert state_to_operators(operators_to_state(XOp())) == XOp()
    raises(NotImplementedError, lambda: operators_to_state(XKet))