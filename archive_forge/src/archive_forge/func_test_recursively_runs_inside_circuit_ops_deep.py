import cirq
def test_recursively_runs_inside_circuit_ops_deep():
    a = cirq.NamedQubit('a')
    small_op = cirq.Z(a) ** 1e-06
    nested_circuit = cirq.FrozenCircuit(cirq.X(a), small_op, small_op.with_tags(NO_COMPILE_TAG), small_op, cirq.Y(a))
    nested_circuit_dropped = cirq.FrozenCircuit(cirq.Moment(cirq.X(a)), cirq.Moment(), cirq.Moment(small_op.with_tags(NO_COMPILE_TAG)), cirq.Moment(), cirq.Moment(cirq.Y(a)))
    c_orig = cirq.Circuit(small_op, cirq.CircuitOperation(nested_circuit).repeat(6).with_tags(NO_COMPILE_TAG), small_op, cirq.CircuitOperation(nested_circuit).repeat(5).with_tags('preserve_tag'), small_op)
    c_expected = cirq.Circuit(cirq.Moment(), cirq.Moment(cirq.CircuitOperation(nested_circuit).repeat(6).with_tags(NO_COMPILE_TAG)), cirq.Moment(), cirq.Moment(cirq.CircuitOperation(nested_circuit_dropped).repeat(5).with_tags('preserve_tag')), cirq.Moment())
    context = cirq.TransformerContext(tags_to_ignore=[NO_COMPILE_TAG], deep=True)
    cirq.testing.assert_same_circuits(cirq.drop_negligible_operations(c_orig, context=context, atol=0.001), c_expected)