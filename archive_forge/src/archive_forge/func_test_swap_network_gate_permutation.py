from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('part_lens, acquaintance_size', list((((part_len,) * n_parts, acquaintance_size) for part_len, acquaintance_size, n_parts in product(range(1, 5), acquaintance_sizes, range(2, 5)))))
def test_swap_network_gate_permutation(part_lens, acquaintance_size):
    n_qubits = sum(part_lens)
    swap_network_gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    cca.testing.assert_permutation_decomposition_equivalence(swap_network_gate, n_qubits)