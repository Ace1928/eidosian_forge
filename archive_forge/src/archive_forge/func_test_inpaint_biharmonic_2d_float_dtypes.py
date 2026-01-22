import numpy as np
from skimage import data, img_as_float
from skimage._shared import testing
from skimage._shared.testing import assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.morphology import binary_dilation, disk
from skimage.restoration import inpaint
@testing.parametrize('dtype', [np.float32, np.float64])
def test_inpaint_biharmonic_2d_float_dtypes(dtype):
    img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    mask = np.zeros_like(img)
    mask[2, 2:] = 1
    mask[1, 3:] = 1
    mask[0, 4:] = 1
    img[np.where(mask)] = 0
    img = img.astype(dtype, copy=False)
    out = inpaint.inpaint_biharmonic(img, mask)
    assert out.dtype == img.dtype
    ref = np.array([[0.0, 0.0625, 0.25, 0.5625, 0.73925058], [0.0, 0.0625, 0.25, 0.5478048, 0.76557821], [0.0, 0.0625, 0.25842878, 0.5623079, 0.85927796], [0.0, 0.0625, 0.25, 0.5625, 1.0], [0.0, 0.0625, 0.25, 0.5625, 1.0]])
    assert_allclose(ref, out, rtol=1e-05)