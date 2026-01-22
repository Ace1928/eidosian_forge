import cirq
def test_clears_known_empties_even_at_zero_tolerance():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.Z(a) ** 0, cirq.Y(a) ** 1e-07, cirq.X(a) ** (-1e-07), cirq.CZ(a, b) ** 0)
    cirq.testing.assert_same_circuits(cirq.drop_negligible_operations(circuit, atol=0.001), cirq.Circuit([cirq.Moment()] * 4))
    cirq.testing.assert_same_circuits(cirq.drop_negligible_operations(circuit, atol=0), cirq.Circuit(cirq.Moment(), cirq.Moment(cirq.Y(a) ** 1e-07), cirq.Moment(cirq.X(a) ** (-1e-07)), cirq.Moment()))