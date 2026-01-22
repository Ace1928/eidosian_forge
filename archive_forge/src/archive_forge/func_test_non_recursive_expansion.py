import cirq
def test_non_recursive_expansion():
    qubits = [cirq.NamedQubit(s) for s in 'xy']
    no_decomp = lambda op: isinstance(op, cirq.GateOperation) and op.gate == cirq.ISWAP
    unexpanded_circuit = cirq.Circuit(cirq.ISWAP(*qubits))
    circuit = cirq.expand_composite(unexpanded_circuit, no_decomp=no_decomp)
    assert circuit == unexpanded_circuit
    no_decomp = lambda op: isinstance(op, cirq.GateOperation) and isinstance(op.gate, (cirq.CNotPowGate, cirq.HPowGate))
    circuit = cirq.expand_composite(unexpanded_circuit, no_decomp=no_decomp)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = '\nx: ───@───H───X───S───X───S^-1───H───@───\n      │       │       │              │\ny: ───X───────@───────@──────────────X───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram