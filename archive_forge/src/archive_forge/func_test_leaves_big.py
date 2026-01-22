import cirq
def test_leaves_big():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.Moment(cirq.Z(a) ** 0.1))
    cirq.testing.assert_same_circuits(cirq.drop_negligible_operations(circuit, atol=0.001), circuit)