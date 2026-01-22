import cirq
def test_do_not_decompose_no_compile():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(q0, q1).with_tags('no_compile'))
    context = cirq.TransformerContext(tags_to_ignore=('no_compile',))
    assert_equal_mod_empty(c, cirq.expand_composite(c, context=context))