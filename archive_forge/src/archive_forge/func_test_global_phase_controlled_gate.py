import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('gate, matrix', [(cirq.ZPowGate(global_shift=-0.5, exponent=1), np.diag([1, 1, -1j, 1j])), (cirq.CZPowGate(global_shift=-0.5, exponent=1), np.diag([1, 1, 1, 1, -1j, -1j, -1j, 1j])), (cirq.XPowGate(global_shift=-0.5, exponent=1), np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), np.array([[0, -1j], [-1j, 0]])]])), (cirq.CXPowGate(global_shift=-0.5, exponent=1), np.block([[np.diag([1, 1, 1, 1, -1j, -1j]), np.zeros((6, 2))], [np.zeros((2, 6)), np.array([[0, -1j], [-1j, 0]])]]))])
def test_global_phase_controlled_gate(gate, matrix):
    np.testing.assert_equal(cirq.unitary(gate.controlled()), matrix)