import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_converts_single_qubit_then_two():
    q0, q1 = cirq.LineQubit.range(2)
    before = cirq.Circuit(cirq.X(q0), cirq.Y(q0), cirq.CZ(q0, q1))
    after = cirq.optimize_for_target_gateset(before, gateset=CliffordTargetGateset(), ignore_failures=False)
    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-07)