import pytest
import numpy as np
from cirq.testing import sample_gates
import cirq
@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 20))
def test_phase_using_clean_ancilla(theta: float):
    g = sample_gates.PhaseUsingCleanAncilla(theta)
    q = cirq.LineQubit(0)
    qubit_order = cirq.QubitOrder.explicit([q], fallback=cirq.QubitOrder.DEFAULT)
    decomposed_unitary = cirq.Circuit(cirq.decompose_once(g.on(q))).unitary(qubit_order=qubit_order)
    phase = np.exp(1j * np.pi * theta)
    np.testing.assert_allclose(g.narrow_unitary(), np.array([[1, 0], [0, phase]]))
    np.testing.assert_allclose(decomposed_unitary, np.array([[1, 0, 0, 0], [0, phase, 0, 0], [0, 0, phase, 0], [0, 0, 0, 1]]))