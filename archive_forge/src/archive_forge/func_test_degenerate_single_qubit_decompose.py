import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_degenerate_single_qubit_decompose():
    q0 = cirq.LineQubit(0)
    before = cirq.Circuit(cirq.Z(q0) ** 0.1, cirq.X(q0) ** 1.0000000001, cirq.Z(q0) ** 0.1)
    expected = cirq.Circuit(cirq.SingleQubitCliffordGate.X(q0))
    after = cirq.optimize_for_target_gateset(before, gateset=CliffordTargetGateset(), ignore_failures=False)
    assert after == expected
    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-07)
    cirq.testing.assert_allclose_up_to_global_phase(after.unitary(), expected.unitary(), atol=1e-07)