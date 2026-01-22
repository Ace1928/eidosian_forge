from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('superoperator, error', ((np.array([[1, 2, 3], [4, 5, 6]]), 'shape'), (np.eye(2), 'shape')))
def test_superoperator_to_choi_invalid_input(superoperator, error):
    with pytest.raises(ValueError, match=error):
        _ = cirq.superoperator_to_choi(superoperator)