import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
def test_invalid_seed():
    seed = np.ones((5, 5))
    mask = np.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed * 2, mask, method='dilation')
    with pytest.raises(ValueError):
        reconstruction(seed * 0.5, mask, method='erosion')