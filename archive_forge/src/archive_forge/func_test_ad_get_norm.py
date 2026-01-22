import numpy as np
import numpy.testing as npt
from .. import rapidart as ra
from ...interfaces.base import Bunch
def test_ad_get_norm():
    params = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, np.pi / 4, np.pi / 4, np.pi / 4, 0, 0, 0, -np.pi / 4, -np.pi / 4, -np.pi / 4]).reshape((3, 6))
    norm, _ = ra._calc_norm(params, False, 'SPM')
    npt.assert_almost_equal(norm, np.array([18.86436316, 37.74610158, 31.29780829]))
    norm, _ = ra._calc_norm(params, True, 'SPM')
    npt.assert_almost_equal(norm, np.array([0.0, 143.72192614, 173.92527131]))