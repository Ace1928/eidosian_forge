import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('eigen_gate_type', [cirq.CZPowGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate])
def test_phase_insensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(eigen_gate_type)