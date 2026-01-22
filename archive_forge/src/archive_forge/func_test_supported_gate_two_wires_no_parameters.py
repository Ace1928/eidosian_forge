from math import sqrt, pi
import pytest
import numpy as np
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('name,expected_output', [('CNOT', [-1 / 2, 1]), ('SWAP', [-1 / 2, -1 / 2]), ('CZ', [-1 / 2, -1 / 2])])
def test_supported_gate_two_wires_no_parameters(self, device, tol, name, expected_output):
    """Tests supported parametrized gates that act on two wires"""
    n_wires = 2
    dev = device(n_wires)
    op = getattr(qml.ops, name)
    if isinstance(dev, qml.Device) and (not dev.supports_operation(op)):
        pytest.skip('operation not supported')

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(np.array([1 / 2, 0, 0, sqrt(3) / 2]), wires=[0, 1])
        op(wires=[0, 1])
        return (qml.expval(qml.Z(0)), qml.expval(qml.Z(1)))
    assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))