import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state_seed():
    state = np.ones(2) / np.sqrt(2)
    samples = cirq.sample_state_vector(state, [0], repetitions=10, seed=1234)
    assert np.array_equal(samples, [[False], [True], [False], [True], [True], [False], [False], [True], [True], [True]])
    samples = cirq.sample_state_vector(state, [0], repetitions=10, seed=np.random.RandomState(1234))
    assert np.array_equal(samples, [[False], [True], [False], [True], [True], [False], [False], [True], [True], [True]])