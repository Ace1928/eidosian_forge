from typing import List
import numpy as np
import pytest
import cirq
def test_merge_1q_unitaries():
    q, q2 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(q) ** 0.5, cirq.Z(q) ** 0.5, cirq.X(q) ** (-0.5))
    c = cirq.merge_k_qubit_unitaries(c, k=1)
    op_list = [*c.all_operations()]
    assert len(op_list) == 1
    assert isinstance(op_list[0].gate, cirq.MatrixGate)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c), cirq.unitary(cirq.Y ** 0.5), atol=1e-07)
    c = cirq.Circuit([cirq.Z(q), cirq.H(q), cirq.X(q), cirq.H(q), cirq.CZ(q, q2), cirq.H(q)])
    c = cirq.drop_empty_moments(cirq.merge_k_qubit_unitaries(c, k=1))
    assert len(c) == 3
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c[0]), np.eye(2), atol=1e-07)
    assert isinstance(c[-1][q].gate, cirq.MatrixGate)