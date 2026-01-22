import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
def test_assert_allclose_up_to_global_phase():
    assert_allclose_up_to_global_phase(np.array([[1]]), np.array([[1j]]), atol=0)
    with pytest.raises(AssertionError):
        assert_allclose_up_to_global_phase(np.array([[1]]), np.array([[2]]), atol=0)
    assert_allclose_up_to_global_phase(np.array([[1e-08, -1, 1e-08]]), np.array([[1e-08, 1, 1e-08]]), atol=1e-06)
    with pytest.raises(AssertionError):
        assert_allclose_up_to_global_phase(np.array([[0.0001, -1, 0.0001]]), np.array([[0.0001, 1, 0.0001]]), atol=1e-06)
    assert_allclose_up_to_global_phase(np.array([[1, 2], [3, 4]]), np.array([[-1, -2], [-3, -4]]), atol=0)