import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_pauliz_hadamard(self, device, tol, skip_if):
    """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
    n_wires = 3
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        skip_if(dev, {'supports_tensor_observables': False})
    theta = 0.432
    phi = 0.123
    varphi = -0.543

    @qml.qnode(dev)
    def circuit():
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.RX(varphi, wires=[2])
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.var(qml.Z(0) @ qml.Hadamard(wires=[1]) @ qml.Y(2))
    res = circuit()
    expected = (3 + np.cos(2 * phi) * np.cos(varphi) ** 2 - np.cos(2 * theta) * np.sin(varphi) ** 2 - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)) / 4
    assert np.allclose(res, expected, atol=tol(dev.shots))