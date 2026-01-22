import numpy as np
import cirq
import pytest
def test_gate_params():
    state = np.array([1, 0, 0, 0], dtype=np.complex64)
    gate = cirq.StatePreparationChannel(state)
    assert gate.num_qubits() == 2
    assert not gate._has_unitary_()
    assert gate._has_kraus_()
    assert str(gate) == 'StatePreparationChannel([1.+0.j 0.+0.j 0.+0.j 0.+0.j])'
    cirq.testing.assert_equivalent_repr(gate)