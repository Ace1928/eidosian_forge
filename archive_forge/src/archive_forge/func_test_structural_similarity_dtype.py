import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64])
def test_structural_similarity_dtype(dtype):
    N = 30
    X = np.random.rand(N, N)
    Y = np.random.rand(N, N)
    if np.dtype(dtype).kind in 'iub':
        data_range = 255.0
        X = (X * 255).astype(np.uint8)
        Y = (X * 255).astype(np.uint8)
    else:
        data_range = 1.0
        X = X.astype(dtype, copy=False)
        Y = Y.astype(dtype, copy=False)
    S1 = structural_similarity(X, Y, data_range=data_range)
    assert S1.dtype == np.float64
    assert S1 < 0.1