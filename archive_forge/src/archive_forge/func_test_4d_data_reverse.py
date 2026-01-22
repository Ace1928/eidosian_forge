import numpy as np
from numpy.testing import (assert_allclose,
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state
def test_4d_data_reverse(self):
    actual = directed_hausdorff(self.path_2_4d, self.path_1_4d)[0]
    expected = max(np.amin(distance.cdist(self.path_1_4d, self.path_2_4d), axis=0))
    assert_allclose(actual, expected)