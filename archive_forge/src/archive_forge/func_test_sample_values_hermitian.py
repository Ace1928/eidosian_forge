import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sample_values_hermitian(self, device, tol):
    """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
    n_wires = 1
    dev = device(n_wires)
    if not dev.shots:
        pytest.skip('Device is in analytic mode, cannot test sampling.')
    if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
        pytest.skip('Skipped because device does not support the Hermitian observable.')
    A_ = np.array([[1, 2j], [-2j, 0]])
    theta = 0.543

    @qml.qnode(dev)
    def circuit():
        qml.RX(theta, wires=[0])
        return qml.sample(qml.Hermitian(A_, wires=0))
    res = circuit().flatten()
    eigvals = np.linalg.eigvalsh(A_)
    assert np.allclose(sorted(list(set(res.tolist()))), sorted(eigvals), atol=tol(dev.shots))
    assert np.allclose(np.mean(res), 2 * np.sin(theta) + 0.5 * np.cos(theta) + 0.5, atol=tol(False))
    assert np.allclose(np.var(res), 0.25 * (np.sin(theta) - 4 * np.cos(theta)) ** 2, atol=tol(False))