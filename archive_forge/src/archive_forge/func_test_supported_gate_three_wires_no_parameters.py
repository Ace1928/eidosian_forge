from math import sqrt, pi
import pytest
import numpy as np
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('name,expected_output', [('CSWAP', [-1, -1, 1])])
def test_supported_gate_three_wires_no_parameters(self, device, tol, name, expected_output):
    """Tests supported non-parametrized gates that act on three wires"""
    n_wires = 3
    dev = device(n_wires)
    op = getattr(qml.ops, name)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
        op(wires=[0, 1, 2])
        return (qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2)))
    assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))