import numpy as np
import pytest
import cirq
def test_product_state():
    q0, q1, q2 = cirq.LineQubit.range(3)
    plus0 = cirq.KET_PLUS(q0)
    plus1 = cirq.KET_PLUS(q1)
    ps = plus0 * plus1
    assert str(plus0) == '+X(q(0))'
    assert str(plus1) == '+X(q(1))'
    assert str(ps) == '+X(q(0)) * +X(q(1))'
    ps *= cirq.KET_ONE(q2)
    assert str(ps) == '+X(q(0)) * +X(q(1)) * -Z(q(2))'
    with pytest.raises(ValueError) as e:
        ps *= cirq.KET_PLUS(q2)
    assert e.match('.*both contain factors for these qubits: \\[cirq.LineQubit\\(2\\)\\]')
    ps2 = eval(repr(ps))
    assert ps == ps2