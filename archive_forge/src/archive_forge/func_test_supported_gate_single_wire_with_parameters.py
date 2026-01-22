from math import sqrt, pi
import pytest
import numpy as np
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('name,par,expected_output', [('PhaseShift', [pi / 2], 1), ('PhaseShift', [-pi / 4], 1), ('RX', [pi / 2], 0), ('RX', [-pi / 4], 1 / sqrt(2)), ('RY', [pi / 2], 0), ('RY', [-pi / 4], 1 / sqrt(2)), ('RZ', [pi / 2], 1), ('RZ', [-pi / 4], 1), ('MultiRZ', [pi / 2], 1), ('MultiRZ', [-pi / 4], 1), ('Rot', [pi / 2, 0, 0], 1), ('Rot', [0, pi / 2, 0], 0), ('Rot', [0, 0, pi / 2], 1), ('Rot', [pi / 2, -pi / 4, -pi / 4], 1 / sqrt(2)), ('Rot', [-pi / 4, pi / 2, pi / 4], 0), ('Rot', [-pi / 4, pi / 4, pi / 2], 1 / sqrt(2)), ('QubitUnitary', [np.array([[1j / sqrt(2), 1j / sqrt(2)], [1j / sqrt(2), -1j / sqrt(2)]])], 0), ('QubitUnitary', [np.array([[-1j / sqrt(2), 1j / sqrt(2)], [1j / sqrt(2), 1j / sqrt(2)]])], 0)])
def test_supported_gate_single_wire_with_parameters(self, device, tol, name, par, expected_output):
    """Tests supported parametrized gates that act on a single wire"""
    n_wires = 1
    dev = device(n_wires)
    op = getattr(qml.ops, name)

    @qml.qnode(dev)
    def circuit():
        op(*par, wires=0)
        return qml.expval(qml.Z(0))
    assert np.isclose(circuit(), expected_output, atol=tol(dev.shots))