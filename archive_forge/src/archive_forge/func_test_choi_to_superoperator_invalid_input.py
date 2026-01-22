from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('choi, error', ((np.array([[1, 2, 3], [4, 5, 6]]), 'shape'), (np.eye(2), 'shape'), (np.array([[0.6, 0.0, -0.1j, 0.1], [0.0, 0.0, 0.0, 0.0], [0.1j, 0.0, 0.4, 0.0], [0.2, 0.0, 0.0, 1.0]]), 'Hermitian')))
def test_choi_to_superoperator_invalid_input(choi, error):
    with pytest.raises(ValueError, match=error):
        _ = cirq.choi_to_superoperator(choi)