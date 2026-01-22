import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_not_square():
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.array([[1], [0]]), num_qubits=1)