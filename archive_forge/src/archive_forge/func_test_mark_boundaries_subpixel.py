import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import find_boundaries, mark_boundaries
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_mark_boundaries_subpixel(dtype):
    labels = np.array([[0, 0, 0, 0], [0, 0, 5, 0], [0, 1, 5, 0], [0, 0, 5, 0], [0, 0, 0, 0]], dtype=np.uint8)
    np.random.seed(0)
    image = np.round(np.random.rand(*labels.shape), 2)
    image = image.astype(dtype, copy=False)
    marked = mark_boundaries(image, labels, color=white, mode='subpixel')
    assert marked.dtype == _supported_float_type(dtype)
    marked_proj = np.round(np.mean(marked, axis=-1), 2)
    ref_result = np.array([[0.55, 0.63, 0.72, 0.69, 0.6, 0.55, 0.54], [0.45, 0.58, 0.72, 1.0, 1.0, 1.0, 0.69], [0.42, 0.54, 0.65, 1.0, 0.44, 1.0, 0.89], [0.69, 1.0, 1.0, 1.0, 0.69, 1.0, 0.83], [0.96, 1.0, 0.38, 1.0, 0.79, 1.0, 0.53], [0.89, 1.0, 1.0, 1.0, 0.38, 1.0, 0.16], [0.57, 0.78, 0.93, 1.0, 0.07, 1.0, 0.09], [0.2, 0.52, 0.92, 1.0, 1.0, 1.0, 0.54], [0.02, 0.35, 0.83, 0.9, 0.78, 0.81, 0.87]])
    assert_allclose(marked_proj, ref_result, atol=0.01)