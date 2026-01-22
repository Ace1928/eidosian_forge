import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_from_state_invalid_state():
    with pytest.raises(ValueError, match='Invalid quantum state'):
        cirq.to_valid_density_matrix(np.array([1, 0, 0]), num_qubits=2)