import pytest
import cirq
def test_complex_circuit():
    """Tests that a complex circuit is correctly optimized."""
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(5)
    input_circuit = cirq.Circuit(cirq.Moment([cirq.X(q1), cirq.ISWAP(q2, q3), cirq.Z(q5)]), cirq.Moment([cirq.X(q1), cirq.ISWAP(q4, q5)]), cirq.Moment([cirq.ISWAP(q1, q2), cirq.X(q4)]))
    expected = cirq.Circuit(cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Z(q5)]), cirq.Moment([cirq.ISWAP(q2, q3), cirq.ISWAP(q4, q5)]), cirq.Moment([cirq.X(q1), cirq.X(q4)]), cirq.Moment([cirq.ISWAP(q1, q2)]))
    cirq.testing.assert_same_circuits(cirq.stratified_circuit(input_circuit, categories=[cirq.X, cirq.Z]), expected)