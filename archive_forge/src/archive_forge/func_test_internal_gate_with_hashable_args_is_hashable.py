import cirq
import cirq_google
import pytest
def test_internal_gate_with_hashable_args_is_hashable():
    hashable = cirq_google.InternalGate(gate_name='GateWithHashableArgs', gate_module='test', num_qubits=3, foo=1, bar='2', baz=(('a', 1),))
    _ = hash(hashable)
    unhashable = cirq_google.InternalGate(gate_name='GateWithHashableArgs', gate_module='test', num_qubits=3, foo=1, bar='2', baz={'a': 1})
    with pytest.raises(TypeError, match='unhashable'):
        _ = hash(unhashable)