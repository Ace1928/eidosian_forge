import cirq
from cirq.contrib.paulistring import clifford_optimized_circuit, CliffordTargetGateset
def test_with_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.X(q0), cirq.CZ(q0, q1), cirq.measure(q0, q1, key='m'))
    c_expected = cirq.optimize_for_target_gateset(cirq.Circuit(cirq.CZ(q0, q1), cirq.X(q0), cirq.Z(q1), cirq.measure(q0, q1, key='m')), gateset=CliffordTargetGateset(), ignore_failures=True)
    c_opt = clifford_optimized_circuit(c_orig)
    cirq.testing.assert_allclose_up_to_global_phase(c_orig.unitary(), c_opt.unitary(), atol=1e-07)
    assert c_opt == c_expected
    cirq.testing.assert_has_diagram(c_opt, "\n0: ───@───X───M('m')───\n      │       │\n1: ───@───Z───M────────\n")