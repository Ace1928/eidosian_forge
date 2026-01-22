import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('p', (0, 0.1, 0.2, 0.5, 0.8, 0.9, 1))
@pytest.mark.parametrize('channel_factory, entanglement_fidelity_formula', ((cirq.depolarize, lambda p: 1 - p), (lambda p: cirq.depolarize(p, n_qubits=2), lambda p: 1 - p), (lambda p: cirq.depolarize(p, n_qubits=3), lambda p: 1 - p), (cirq.amplitude_damp, lambda gamma: 1 / 2 - gamma / 4 + np.sqrt(1 - gamma) / 2)))
def test_entanglement_fidelity_of_noisy_channels(p, channel_factory, entanglement_fidelity_formula):
    channel = channel_factory(p)
    actual_entanglement_fidelity = cirq.entanglement_fidelity(channel)
    expected_entanglement_fidelity = entanglement_fidelity_formula(p)
    assert np.isclose(actual_entanglement_fidelity, expected_entanglement_fidelity)