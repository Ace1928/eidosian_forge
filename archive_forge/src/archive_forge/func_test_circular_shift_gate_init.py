import cirq
import cirq.contrib.acquaintance as cca
def test_circular_shift_gate_init():
    g = cca.CircularShiftGate(4, 2)
    assert g.num_qubits() == 4
    assert g.shift == 2
    g = cca.CircularShiftGate(4, 1, swap_gate=cirq.CZ)
    assert g.swap_gate == cirq.CZ