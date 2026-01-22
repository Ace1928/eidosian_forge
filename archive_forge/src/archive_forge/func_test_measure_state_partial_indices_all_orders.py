import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('use_np_transpose', [False, True])
def test_measure_state_partial_indices_all_orders(use_np_transpose: bool):
    linalg.can_numpy_support_shape = lambda s: use_np_transpose
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            initial_state = cirq.to_valid_state_vector(x, 3)
            bits, state = cirq.measure_state_vector(initial_state, perm)
            np.testing.assert_almost_equal(state, initial_state)
            assert bits == [bool(1 & x >> 2 - p) for p in perm]