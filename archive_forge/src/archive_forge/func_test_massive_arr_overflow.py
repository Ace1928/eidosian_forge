import numpy as np
from numpy.testing import (assert_allclose,
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state
@pytest.mark.xslow
def test_massive_arr_overflow():
    try:
        import psutil
    except ModuleNotFoundError:
        pytest.skip('psutil required to check available memory')
    if psutil.virtual_memory().available < 80 * 2 ** 30:
        pytest.skip('insufficient memory available to run this test')
    size = int(3000000000.0)
    arr1 = np.zeros(shape=(size, 2))
    arr2 = np.zeros(shape=(3, 2))
    arr1[size - 1] = [5, 5]
    actual = directed_hausdorff(u=arr1, v=arr2)
    assert_allclose(actual[0], 7.0710678118654755)
    assert_allclose(actual[1], size - 1)