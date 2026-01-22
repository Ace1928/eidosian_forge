import numpy as np
import numpy.linalg as npl
import pytest
from numpy.testing import assert_almost_equal
from ..affines import apply_affine, from_matvec
from ..eulerangles import euler2mat
from ..nifti1 import Nifti1Image
from ..spaces import slice2volume, vox2out_vox
def get_outspace_params():
    trans_123 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    trans_m123 = [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]
    rot_3 = from_matvec(euler2mat(np.pi / 4), [0, 0, 0])
    return (((2, 3, 4), np.eye(4), None, (2, 3, 4), np.eye(4)), ((2, 3, 4), np.diag([-1, 1, 1, 1]), None, (2, 3, 4), [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), ((2, 3, 4), np.diag([4, 5, 6, 1]), None, (5, 11, 19), np.eye(4)), ((2, 3, 4), np.diag([4, 5, 6, 1]), (4, 5, 6), (2, 3, 4), np.diag([4, 5, 6, 1])), ((2, 3, 4), trans_123, None, (2, 3, 4), trans_123), ((2, 3, 4), trans_m123, None, (2, 3, 4), trans_m123), ((2, 3, 4), rot_3, None, (4, 4, 4), [[1, 0, 0, -2 * np.cos(np.pi / 4)], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), ((2, 3), np.eye(4), None, (2, 3), np.eye(4)), ((2,), np.eye(4), None, (2,), np.eye(4)), ((2, 3), np.diag([4, 5, 6, 1]), (4, 5), (2, 3), np.diag([4, 5, 1, 1])))