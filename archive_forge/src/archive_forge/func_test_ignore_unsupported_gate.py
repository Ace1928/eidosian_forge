import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_ignore_unsupported_gate():

    class UnsupportedGate(cirq.testing.TwoQubitGate):
        pass
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(UnsupportedGate()(q0, q1), cirq.X(q0) ** sympy.Symbol('theta'))
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=CliffordTargetGateset(), ignore_failures=True)
    assert c_new == c_orig