import cirq
from cirq.contrib.paulistring import clifford_optimized_circuit, CliffordTargetGateset
def test_remove_staggered_czs():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(cirq.CZ(q0, q1), cirq.CZ(q1, q2), cirq.CZ(q0, q1))
    c_expected = cirq.optimize_for_target_gateset(cirq.Circuit(cirq.CZ(q1, q2)), gateset=CliffordTargetGateset(), ignore_failures=True)
    c_opt = clifford_optimized_circuit(c_orig)
    cirq.testing.assert_allclose_up_to_global_phase(c_orig.unitary(), c_opt.unitary(qubits_that_should_be_present=(q0, q1, q2)), atol=1e-07)
    assert c_opt == c_expected
    cirq.testing.assert_has_diagram(c_opt, '\n1: ───@───\n      │\n2: ───@───\n')