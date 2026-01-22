import numpy as np
import numpy.testing as npt
from .. import rapidart as ra
from ...interfaces.base import Bunch
def test_ad_get_affine_matrix():
    matrix = ra._get_affine_matrix(np.array([0]), 'SPM')
    npt.assert_equal(matrix, np.eye(4))
    params = [1, 2, 3]
    matrix = ra._get_affine_matrix(params, 'SPM')
    out = np.eye(4)
    out[0:3, 3] = params
    npt.assert_equal(matrix, out)
    params = np.array([0, 0, 0, np.pi / 2, np.pi / 2, np.pi / 2])
    matrix = ra._get_affine_matrix(params, 'SPM')
    out = np.array([0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape((4, 4))
    npt.assert_almost_equal(matrix, out)
    params = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3])
    matrix = ra._get_affine_matrix(params, 'SPM')
    out = np.array([1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1]).reshape((4, 4))
    npt.assert_equal(matrix, out)
    params = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 3])
    matrix = ra._get_affine_matrix(params, 'SPM')
    out = np.array([1, 1, 2, 0, 0, 1, 3, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape((4, 4))
    npt.assert_equal(matrix, out)