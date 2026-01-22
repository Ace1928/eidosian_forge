import cirq
def test_no_move():
    q1 = cirq.NamedQubit('q1')
    before = cirq.Circuit([cirq.Moment([cirq.H(q1)])])
    after = before
    assert_optimizes(before=before, after=after)