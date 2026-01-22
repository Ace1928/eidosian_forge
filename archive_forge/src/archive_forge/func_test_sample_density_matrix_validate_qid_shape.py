import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_validate_qid_shape():
    matrix = cirq.to_valid_density_matrix(0, 3)
    cirq.sample_density_matrix(matrix, [], qid_shape=(2, 2, 2))
    with pytest.raises(ValueError, match='Matrix size does not match qid shape'):
        cirq.sample_density_matrix(matrix, [], qid_shape=(2, 2, 1))
    matrix2 = cirq.to_valid_density_matrix(0, qid_shape=(1, 2, 3))
    cirq.sample_density_matrix(matrix2, [], qid_shape=(1, 2, 3))
    with pytest.raises(ValueError, match='Matrix size does not match qid shape'):
        cirq.sample_density_matrix(matrix2, [], qid_shape=(2, 2, 2))