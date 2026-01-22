import cirq
def test_does_not_clear_small_no_compile():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.Moment((cirq.Z(a) ** 1e-06).with_tags(NO_COMPILE_TAG)))
    cirq.testing.assert_same_circuits(cirq.drop_negligible_operations(circuit, context=cirq.TransformerContext(tags_to_ignore=(NO_COMPILE_TAG,)), atol=0.001), circuit)