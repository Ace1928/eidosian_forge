import numpy as np
import pytest
import cirq
def test_fidelity_numpy_arrays():
    vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64)
    vec2 = np.array([1, 0, 0, 0], dtype=np.complex64)
    tensor1 = np.reshape(vec1, (2, 2, 2))
    tensor2 = np.reshape(vec2, (2, 2))
    mat1 = np.outer(vec1, vec1.conj())
    np.testing.assert_allclose(cirq.fidelity(vec1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, tensor1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor1, vec1), 1)
    np.testing.assert_allclose(cirq.fidelity(vec1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor1, mat1), 1)
    np.testing.assert_allclose(cirq.fidelity(tensor2, tensor2, qid_shape=(2, 2)), 1)
    np.testing.assert_allclose(cirq.fidelity(mat1, mat1, qid_shape=(8,)), 1)
    with pytest.raises(ValueError, match='dimension'):
        _ = cirq.fidelity(vec1, vec1, qid_shape=(4,))
    with pytest.raises(ValueError, match='Mismatched'):
        _ = cirq.fidelity(vec1, vec2)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.fidelity(tensor2, tensor2)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.fidelity(mat1, mat1)