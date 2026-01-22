from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('part_lens, acquaintance_size', part_lens_and_acquaintance_sizes)
def test_swap_network_permutation(part_lens, acquaintance_size):
    n_qubits = sum(part_lens)
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    expected_permutation = {i: j for i, j in zip(range(n_qubits), reversed(range(n_qubits)))}
    assert gate.permutation() == expected_permutation