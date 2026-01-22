import cirq
def test_no_slide_forward_one():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    before = cirq.Circuit([cirq.Moment([cirq.H(q1), cirq.measure(q2), cirq.measure(q3)])])
    after = cirq.Circuit([cirq.Moment([cirq.H(q1), cirq.measure(q2), cirq.measure(q3)])])
    assert_optimizes(before=before, after=after, measure_only_moment=False)