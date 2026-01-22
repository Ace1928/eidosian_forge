import cirq
import cirq.contrib.acquaintance as cca
def test_circular_shift_gate_repr():
    g = cca.CircularShiftGate(3, 2)
    cirq.testing.assert_equivalent_repr(g)