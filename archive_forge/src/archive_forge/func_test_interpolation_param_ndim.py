import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('t', [[[0, 0.5]], [[[[[[[[[0, 0.5]]]]]]]]]])
def test_interpolation_param_ndim(self, t):
    arr1 = np.array([0, 1])
    arr2 = np.array([1, 0])
    with pytest.raises(ValueError):
        geometric_slerp(start=arr1, end=arr2, t=t)
    with pytest.raises(ValueError):
        geometric_slerp(start=arr1, end=arr1, t=t)