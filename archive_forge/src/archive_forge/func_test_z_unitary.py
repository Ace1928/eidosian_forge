import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_z_unitary():
    assert np.allclose(cirq.unitary(cirq.Z), np.array([[1, 0], [0, -1]]))
    assert np.allclose(cirq.unitary(cirq.Z ** 0.5), np.array([[1, 0], [0, 1j]]))
    assert np.allclose(cirq.unitary(cirq.Z ** 0), np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.unitary(cirq.Z ** (-0.5)), np.array([[1, 0], [0, -1j]]))