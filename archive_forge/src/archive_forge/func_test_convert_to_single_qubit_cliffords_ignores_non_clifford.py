import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_convert_to_single_qubit_cliffords_ignores_non_clifford():
    q0 = cirq.LineQubit(0)
    c_orig = cirq.Circuit(cirq.Z(q0) ** 0.25)
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=CliffordTargetGateset(single_qubit_target=CliffordTargetGateset.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS), ignore_failures=True)
    assert c_orig == c_new