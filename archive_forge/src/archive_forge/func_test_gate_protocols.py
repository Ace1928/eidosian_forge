import numpy as np
import pytest
import sympy
import cirq
def test_gate_protocols():
    for p in [1, 1j, -1]:
        cirq.testing.assert_implements_consistent_protocols(cirq.GlobalPhaseGate(p))
    np.testing.assert_allclose(cirq.unitary(cirq.GlobalPhaseGate(1j)), np.array([[1j]]), atol=1e-08)