import numpy as np
import numpy.linalg as npl
import pytest
from numpy.testing import assert_almost_equal
from ..affines import apply_affine, from_matvec
from ..eulerangles import euler2mat
from ..nifti1 import Nifti1Image
from ..spaces import slice2volume, vox2out_vox
def test_slice2volume():
    for axis, def_aff in zip((0, 1, 2), ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])):
        for val in (0, 5, 10):
            exp_aff = np.array(def_aff)
            exp_aff[axis, -1] = val
            assert (slice2volume(val, axis) == exp_aff).all()