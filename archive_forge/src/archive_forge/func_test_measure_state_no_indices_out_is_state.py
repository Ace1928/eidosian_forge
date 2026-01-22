import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('use_np_transpose', [False, True])
def test_measure_state_no_indices_out_is_state(use_np_transpose: bool):
    linalg.can_numpy_support_shape = lambda s: use_np_transpose
    initial_state = cirq.to_valid_state_vector(0, 3)
    bits, state = cirq.measure_state_vector(initial_state, [], out=initial_state)
    assert [] == bits
    np.testing.assert_almost_equal(state, initial_state)
    assert state is initial_state