import cirq
import cirq_google
import pytest
def test_internal_gate_with_no_args():
    g = cirq_google.InternalGate(gate_name='GateWithNoArgs', gate_module='test', num_qubits=3)
    assert str(g) == 'test.GateWithNoArgs()'
    want_repr = "cirq_google.InternalGate(gate_name='GateWithNoArgs', gate_module='test', num_qubits=3)"
    assert repr(g) == want_repr
    assert cirq.qid_shape(g) == (2, 2, 2)