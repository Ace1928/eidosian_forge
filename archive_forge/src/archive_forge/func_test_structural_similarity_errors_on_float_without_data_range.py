import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_structural_similarity_errors_on_float_without_data_range(dtype):
    X = np.zeros((64, 64), dtype=dtype)
    with pytest.raises(ValueError):
        structural_similarity(X, X)