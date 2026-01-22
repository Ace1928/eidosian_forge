import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_expand_to():
    a, b = cirq.LineQubit.range(2)
    m1 = cirq.Moment(cirq.H(a))
    m2 = m1.expand_to({a})
    assert m1 == m2
    m3 = m1.expand_to({a, b})
    assert m1 != m3
    assert m3.qubits == {a, b}
    assert m3.operations == (cirq.H(a), cirq.I(b))
    with pytest.raises(ValueError, match='superset'):
        _ = m1.expand_to({b})