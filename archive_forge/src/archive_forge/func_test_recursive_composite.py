import cirq
def test_recursive_composite():
    q0, q1 = cirq.LineQubit.range(2)
    swap = cirq.SWAP(q0, q1)
    circuit = cirq.Circuit()
    circuit.append(swap)
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit(cirq.Y(q1) ** (-0.5), cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5, cirq.Y(q0) ** (-0.5), cirq.CZ(q1, q0), cirq.Y(q0) ** 0.5, cirq.Y(q1) ** (-0.5), cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5)
    assert_equal_mod_empty(expected, circuit)