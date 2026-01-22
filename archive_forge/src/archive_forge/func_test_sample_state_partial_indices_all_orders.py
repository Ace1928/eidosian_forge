import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state_partial_indices_all_orders():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            state = cirq.to_valid_state_vector(x, 3)
            expected = [[bool(1 & x >> 2 - p) for p in perm]]
            np.testing.assert_equal(cirq.sample_state_vector(state, perm), expected)