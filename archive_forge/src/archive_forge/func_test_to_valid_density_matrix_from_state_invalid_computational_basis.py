import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_from_state_invalid_computational_basis():
    with pytest.raises(ValueError, match='out of range'):
        cirq.to_valid_density_matrix(-1, num_qubits=2)