import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_converts_single_qubit_series():
    q0 = cirq.LineQubit(0)
    before = cirq.Circuit(cirq.X(q0), cirq.Y(q0), cirq.Z(q0), cirq.X(q0) ** 0.5, cirq.Y(q0) ** 0.5, cirq.Z(q0) ** 0.5, cirq.X(q0) ** (-0.5), cirq.Y(q0) ** (-0.5), cirq.Z(q0) ** (-0.5), cirq.X(q0) ** 0.25, cirq.Y(q0) ** 0.25, cirq.Z(q0) ** 0.25)
    after = cirq.optimize_for_target_gateset(before, gateset=CliffordTargetGateset(), ignore_failures=False)
    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-07)