import cirq
def test_composite_default():
    q0, q1 = cirq.LineQubit.range(2)
    cnot = cirq.CNOT(q0, q1)
    circuit = cirq.Circuit()
    circuit.append(cnot)
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit()
    expected.append([cirq.Y(q1) ** (-0.5), cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5])
    assert_equal_mod_empty(expected, circuit)