import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_invalid_subspaces():
    with pytest.raises(ValueError, match='Subspace specified does not exist in axis'):
        _ = cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((2,), dtype=np.complex64), available_buffer=cirq.eye_tensor((2,), dtype=np.complex64), axes=(0,), subspaces=[(1, 2)])
    with pytest.raises(ValueError, match='Subspace count does not match axis count'):
        _ = cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((2,), dtype=np.complex64), available_buffer=cirq.eye_tensor((2,), dtype=np.complex64), axes=(0,), subspaces=[(0, 1), (0, 1)])
    with pytest.raises(ValueError, match='has zero dimensions'):
        _ = cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((2,), dtype=np.complex64), available_buffer=cirq.eye_tensor((2,), dtype=np.complex64), axes=(0,), subspaces=[()])
    with pytest.raises(ValueError, match='does not have consistent step size'):
        _ = cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3,), dtype=np.complex64), available_buffer=cirq.eye_tensor((3,), dtype=np.complex64), axes=(0,), subspaces=[(0, 2, 1)])