import cirq
def test_infinitely_fast():
    assert cirq.UNCONSTRAINED_DEVICE.duration_of(cirq.X(cirq.NamedQubit('a'))) == cirq.Duration(picos=0)