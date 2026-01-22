import cirq
def test_align_right_no_compile_context():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    cirq.testing.assert_same_circuits(cirq.align_right(cirq.Circuit([cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Y(q1), cirq.X(q2).with_tags('nocompile')]), cirq.Moment([cirq.X(q1), cirq.Y(q2)]), cirq.Moment([cirq.Y(q1)]), cirq.measure(*[q1, q2], key='a')]), context=cirq.TransformerContext(tags_to_ignore=['nocompile'])), cirq.Circuit([cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Y(q1), cirq.X(q2).with_tags('nocompile')]), cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Y(q1), cirq.Y(q2)]), cirq.measure(*[q1, q2], key='a')]))