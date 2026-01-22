import numpy as np
from skimage import data, img_as_float
from skimage._shared import testing
from skimage._shared.testing import assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.morphology import binary_dilation, disk
from skimage.restoration import inpaint
@testing.parametrize('channel_axis', [0, 1, -1])
def test_inpaint_biharmonic_2d_color(channel_axis):
    img = img_as_float(data.astronaut()[:64, :64])
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[8:16, :16] = 1
    img_defect = img * ~mask[..., np.newaxis]
    mse_defect = mean_squared_error(img, img_defect)
    img_defect = np.moveaxis(img_defect, -1, channel_axis)
    img_restored = inpaint.inpaint_biharmonic(img_defect, mask, channel_axis=channel_axis)
    img_restored = np.moveaxis(img_restored, channel_axis, -1)
    mse_restored = mean_squared_error(img, img_restored)
    assert mse_restored < 0.01 * mse_defect