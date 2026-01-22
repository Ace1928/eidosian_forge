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
def test_swap_network_repr(part_lens, acquaintance_size):
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    ct.assert_equivalent_repr(gate)