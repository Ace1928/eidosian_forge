import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_out_broadcast_error():
    with pytest.raises(ValueError):
        np.add(np.array([1, 2, 3]), np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError):
        out = np.array([0])
        np.add(np.array([[1, 2], [3, 4]]), np.array([1, 2]), out=out)
    with pytest.raises(ValueError):
        out = np.array([0, 0])
        np.add(np.array([[1, 2], [3, 4]]), np.array([1, 2]), out=out)
    with pytest.raises(ValueError):
        out = np.array([0, 0])
        np.add(np.array([[1, 2]]), np.array([1, 2]), out=out)
    with pytest.raises(ValueError):
        out = np.array([[0, 0], [0, 0], [0, 0]])
        np.add(np.array([[1, 2], [3, 4]]), np.array([1, 2]), out=out)