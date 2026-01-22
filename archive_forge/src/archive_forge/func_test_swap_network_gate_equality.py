from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('part_len_sets', [set((tuple((randint(1, 5) for _ in range(randint(2, 7)))) for _ in range(5)))])
def test_swap_network_gate_equality(part_len_sets):
    acquaintance_sizes = [None, 0, 1, 2, 3]
    swap_gates = [cirq.SWAP, cirq.CNOT]
    equals_tester = ct.EqualsTester()
    for args in product(part_len_sets, acquaintance_sizes, swap_gates):
        first_gate = cca.SwapNetworkGate(*args)
        second_gate = cca.SwapNetworkGate(*args)
        equals_tester.add_equality_group(first_gate, second_gate)