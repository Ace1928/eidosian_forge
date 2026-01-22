import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_var_hermitian(self, device, tol):
    """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
    n_wires = 2
    dev = device(n_wires)
    if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
        pytest.skip('Skipped because device does not support the Hermitian observable.')
    phi = 0.543
    theta = 0.6543
    H = 0.1 * np.array([[4, -1 + 6j], [-1 - 6j, 2]])

    @qml.qnode(dev)
    def circuit():
        qml.RX(phi, wires=[0])
        qml.RY(theta, wires=[0])
        return qml.var(qml.Hermitian(H, wires=0))
    res = circuit()
    expected = 0.01 * 0.5 * (2 * np.sin(2 * theta) * np.cos(phi) ** 2 + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta)) + 35 * np.cos(2 * phi) + 39)
    assert np.allclose(res, expected, atol=tol(dev.shots))