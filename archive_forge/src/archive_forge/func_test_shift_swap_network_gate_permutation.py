import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('left_part_lens,right_part_lens', [tuple((random_part_lens(2, 2) for _ in ('left', 'right'))) for _ in range(5)])
def test_shift_swap_network_gate_permutation(left_part_lens, right_part_lens):
    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens)
    n_qubits = gate.qubit_count()
    cca.testing.assert_permutation_decomposition_equivalence(gate, n_qubits)