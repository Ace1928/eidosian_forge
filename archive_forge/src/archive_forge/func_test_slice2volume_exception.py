import numpy as np
import numpy.linalg as npl
import pytest
from numpy.testing import assert_almost_equal
from ..affines import apply_affine, from_matvec
from ..eulerangles import euler2mat
from ..nifti1 import Nifti1Image
from ..spaces import slice2volume, vox2out_vox
@pytest.mark.parametrize('index, axis', [[-1, 0], [0, -1], [0, 3]])
def test_slice2volume_exception(index, axis):
    with pytest.raises(ValueError):
        slice2volume(index, axis)