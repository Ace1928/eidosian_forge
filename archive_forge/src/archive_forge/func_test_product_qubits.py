import numpy as np
import pytest
import cirq
def test_product_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    ps = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ZERO(q2)
    assert ps.qubits == [q0, q1, q2]
    assert ps[q0] == cirq.KET_PLUS