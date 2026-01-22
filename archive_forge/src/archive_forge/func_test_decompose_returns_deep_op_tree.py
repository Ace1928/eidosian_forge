import cirq
def test_decompose_returns_deep_op_tree():

    class ExampleGate(cirq.testing.TwoQubitGate):

        def _decompose_(self, qubits):
            q0, q1 = qubits
            yield ((cirq.X(q0), cirq.Y(q0)), cirq.Z(q0))
            yield [cirq.X(q0), [cirq.Y(q0), cirq.Z(q0)]]

            def generator(depth):
                if depth <= 0:
                    yield (cirq.CZ(q0, q1), cirq.Y(q0))
                else:
                    yield (cirq.X(q0), generator(depth - 1))
                    yield cirq.Z(q0)
            yield generator(2)
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(ExampleGate()(q0, q1))
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit(cirq.X(q0), cirq.Y(q0), cirq.Z(q0), cirq.X(q0), cirq.Y(q0), cirq.Z(q0), cirq.X(q0), cirq.X(q0), cirq.CZ(q0, q1), cirq.Y(q0), cirq.Z(q0), cirq.Z(q0))
    assert_equal_mod_empty(expected, circuit)