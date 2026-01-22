import cirq
import cirq_google
import pytest
def test_internal_gate():
    g = cirq_google.InternalGate(gate_name='CouplerDelayZ', gate_module='internal_module', num_qubits=2, delay=1, zpa=0.0, zpl=None)
    assert str(g) == 'internal_module.CouplerDelayZ(delay=1, zpa=0.0, zpl=None)'
    want_repr = "cirq_google.InternalGate(gate_name='CouplerDelayZ', gate_module='internal_module', num_qubits=2, delay=1, zpa=0.0, zpl=None)"
    assert repr(g) == want_repr
    assert cirq.qid_shape(g) == (2, 2)