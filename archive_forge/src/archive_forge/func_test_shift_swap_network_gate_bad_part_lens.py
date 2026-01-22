import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
def test_shift_swap_network_gate_bad_part_lens():
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((0, 1, 1), (2, 2))
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((-1, 1, 1), (2, 2))
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((1, 1), (2, 0, 2))
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((1, 1), (2, -3))