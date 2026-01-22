import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('mat', [np.diag([1]), np.diag([1j]), np.diag([-1]), np.eye(4), np.array([[1.0000000000000002 + 0j, 0, 0, 4.266421588589642e-17j], [0, 1.0000000000000002, -4.266421588589642e-17j, 0], [0, 4.266421588589642e-17j, 1.0000000000000002, 0], [-4.266421588589642e-17j, 0, 0, 1.0000000000000002]]), SWAP, CNOT, Y, H, cirq.kron(H, H), cirq.kron(Y, Y), QFT] + [cirq.testing.random_unitary(2) for _ in range(10)] + [cirq.testing.random_unitary(4) for _ in range(10)] + [cirq.testing.random_unitary(k) for k in range(1, 10)])
def test_bidiagonalize_unitary_with_special_orthogonals(mat):
    p, d, q = cirq.bidiagonalize_unitary_with_special_orthogonals(mat)
    assert cirq.is_special_orthogonal(p)
    assert cirq.is_special_orthogonal(q)
    assert np.allclose(p.dot(mat).dot(q), np.diag(d))
    assert_bidiagonalized_by(mat, p, q)