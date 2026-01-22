import cirq
def test_ignores_large_ops():
    qnum = 20
    qubits = cirq.LineQubit.range(qnum)
    subcircuit = cirq.FrozenCircuit(cirq.X.on_each(*qubits))
    circuit = cirq.Circuit(cirq.CircuitOperation(subcircuit).repeat(10), cirq.measure(*qubits, key='out'))
    cirq.testing.assert_same_circuits(circuit, cirq.drop_negligible_operations(circuit, context=cirq.TransformerContext(deep=True)))