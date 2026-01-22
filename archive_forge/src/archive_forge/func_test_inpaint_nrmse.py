import numpy as np
from skimage import data, img_as_float
from skimage._shared import testing
from skimage._shared.testing import assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.morphology import binary_dilation, disk
from skimage.restoration import inpaint
@testing.parametrize('dtype', [np.uint8, np.float32, np.float64])
@testing.parametrize('order', ['C', 'F'])
@testing.parametrize('channel_axis', [None, -1])
@testing.parametrize('split_into_regions', [False, True])
def test_inpaint_nrmse(dtype, order, channel_axis, split_into_regions):
    image_orig = data.astronaut()[:, :200]
    float_dtype = np.float32 if dtype == np.float32 else np.float64
    image_orig = image_orig.astype(float_dtype, copy=False)
    mask = np.zeros(image_orig.shape[:-1], dtype=bool)
    mask[20:50, 3:20] = 1
    mask[165:180, 90:155] = 1
    mask[40:60, 170:195] = 1
    mask[-60:-40, 170:195] = 1
    mask[-180:-165, 90:155] = 1
    mask[-50:-20, :20] = 1
    mask[200:205, -200:] = 1
    mask[150:255, 20:22] = 1
    mask[365:368, 60:130] = 1
    rstate = np.random.default_rng(0)
    for radius in [0, 2, 4]:
        thresh = 3.25 + 0.25 * radius
        tmp_mask = rstate.standard_normal(image_orig.shape[:-1]) > thresh
        if radius > 0:
            tmp_mask = binary_dilation(tmp_mask, disk(radius, dtype=bool))
        mask[tmp_mask] = 1
    image_defect = image_orig.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(mask)] = 0
    if channel_axis is None:
        image_orig = rgb2gray(image_orig)
        image_defect = rgb2gray(image_defect)
    image_orig = image_orig.astype(dtype, copy=False)
    image_defect = image_defect.astype(dtype, copy=False)
    image_defect = np.asarray(image_defect, order=order)
    image_result = inpaint.inpaint_biharmonic(image_defect, mask, channel_axis=channel_axis, split_into_regions=split_into_regions)
    assert image_result.dtype == float_dtype
    nrmse_defect = normalized_root_mse(image_orig, image_defect)
    nrmse_result = normalized_root_mse(img_as_float(image_orig), image_result)
    assert nrmse_result < 0.2 * nrmse_defect