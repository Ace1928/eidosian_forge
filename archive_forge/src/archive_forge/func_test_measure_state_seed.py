import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('use_np_transpose', [False, True])
def test_measure_state_seed(use_np_transpose: bool):
    linalg.can_numpy_support_shape = lambda s: use_np_transpose
    n = 10
    initial_state = np.ones(2 ** n) / 2 ** (n / 2)
    bits, state1 = cirq.measure_state_vector(initial_state, range(n), seed=1234)
    np.testing.assert_equal(bits, [False, False, True, True, False, False, False, True, False, False])
    bits, state2 = cirq.measure_state_vector(initial_state, range(n), seed=np.random.RandomState(1234))
    np.testing.assert_equal(bits, [False, False, True, True, False, False, False, True, False, False])
    np.testing.assert_allclose(state1, state2)