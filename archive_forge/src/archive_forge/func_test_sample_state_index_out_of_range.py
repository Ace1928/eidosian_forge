import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state_index_out_of_range():
    state = cirq.to_valid_state_vector(0, 3)
    with pytest.raises(IndexError, match='-2'):
        cirq.sample_state_vector(state, [-2])
    with pytest.raises(IndexError, match='3'):
        cirq.sample_state_vector(state, [3])