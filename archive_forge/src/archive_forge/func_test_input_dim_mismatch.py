import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start, end', [(np.zeros(7), np.ones(3)), (np.zeros(2), np.ones(1)), (np.array([]), np.ones(3))])
def test_input_dim_mismatch(self, start, end):
    with pytest.raises(ValueError, match='dimensions'):
        geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))