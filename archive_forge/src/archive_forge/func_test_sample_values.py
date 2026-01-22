import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sample_values(self, device, tol):
    """Tests if the samples returned by sample have
        the correct values
        """
    n_wires = 1
    dev = device(n_wires)
    if not dev.shots:
        pytest.skip('Device is in analytic mode, cannot test sampling.')

    @qml.qnode(dev)
    def circuit():
        qml.RX(1.5708, wires=[0])
        return qml.sample(qml.Z(0))
    res = circuit()
    assert np.allclose(res ** 2, 1, atol=tol(False))