import cirq
def test_slide_forward_one():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    before = cirq.Circuit([cirq.Moment([cirq.H(q1), cirq.measure(q2).with_tags(NO_COMPILE_TAG), cirq.measure(q3)])])
    after = cirq.Circuit([cirq.Moment([cirq.H(q1)]), cirq.Moment([cirq.measure(q2).with_tags(NO_COMPILE_TAG), cirq.measure(q3)])])
    after_no_compile = cirq.Circuit([cirq.Moment([cirq.H(q1), cirq.measure(q2).with_tags(NO_COMPILE_TAG)]), cirq.Moment([cirq.measure(q3)])])
    assert_optimizes(before=before, after=after)
    assert_optimizes(before=before, after=after_no_compile, with_context=True)