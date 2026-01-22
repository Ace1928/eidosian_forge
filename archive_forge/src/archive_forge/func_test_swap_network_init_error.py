from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_swap_network_init_error():
    with pytest.raises(ValueError):
        cca.SwapNetworkGate(())
    with pytest.raises(ValueError):
        cca.SwapNetworkGate((3,))