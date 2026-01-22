import cirq
import pytest
def test_multi_qubit_gate_inputs():
    device = cirq.testing.construct_grid_device(4, 4)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    q = cirq.LineQubit.range(5)
    invalid_subcircuit_op = cirq.CircuitOperation(cirq.Circuit(cirq.X(q[1]), cirq.CCZ(q[0], q[1], q[2]), cirq.Y(q[1])).freeze()).with_tags('<mapped_circuit_op>')
    invalid_circuit = cirq.Circuit(cirq.H(q[0]), cirq.H(q[2]), invalid_subcircuit_op)
    with pytest.raises(ValueError, match='Input circuit must only have ops that act on 1 or 2 qubits.'):
        router(invalid_circuit, context=cirq.TransformerContext(deep=True))
    with pytest.raises(ValueError, match='Input circuit must only have ops that act on 1 or 2 qubits.'):
        router(invalid_circuit, context=cirq.TransformerContext(deep=False))
    invalid_circuit = cirq.Circuit(cirq.CCX(q[0], q[1], q[2]))
    with pytest.raises(ValueError, match='Input circuit must only have ops that act on 1 or 2 qubits.'):
        router(invalid_circuit, context=cirq.TransformerContext(deep=True))
    with pytest.raises(ValueError, match='Input circuit must only have ops that act on 1 or 2 qubits.'):
        router(invalid_circuit, context=cirq.TransformerContext(deep=False))
    valid_subcircuit_op = cirq.CircuitOperation(cirq.Circuit(cirq.X(q[1]), cirq.CZ(q[0], q[1]), cirq.CZ(q[1], q[2]), cirq.Y(q[1])).freeze()).with_tags('<mapped_circuit_op>')
    valid_circuit = cirq.Circuit(cirq.H(q[0]), cirq.H(q[2]), valid_subcircuit_op)
    with pytest.raises(ValueError, match='Input circuit must only have ops that act on 1 or 2 qubits.'):
        router(invalid_circuit, context=cirq.TransformerContext(deep=False))
    c_routed = router(valid_circuit, context=cirq.TransformerContext(deep=True))
    device.validate_circuit(c_routed)