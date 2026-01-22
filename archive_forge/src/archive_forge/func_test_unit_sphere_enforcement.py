import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start, end', [(np.array([1, 0]), np.array([0, 0])), (np.array([1 + 1e-06, 0, 0]), np.array([0, 1 - 1e-06, 0])), (np.array([1 + 1e-06, 0, 0, 0]), np.array([0, 1 - 1e-06, 0, 0]))])
def test_unit_sphere_enforcement(self, start, end):
    with pytest.raises(ValueError, match='unit n-sphere'):
        geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 5))