import numpy as np
from skimage._shared.testing import assert_almost_equal, assert_equal
from skimage import data, img_as_float
from skimage.morphology import diamond
from skimage.feature import match_template, peak_local_max
from skimage._shared import testing
def test_pad_input():
    """Test `match_template` when `pad_input=True`.

    This test places two full templates (one with values lower than the image
    mean, the other higher) and two half templates, which are on the edges of
    the image. The two full templates should score the top (positive and
    negative) matches and the centers of the half templates should score 2nd.
    """
    template = 0.5 * diamond(2)
    image = 0.5 * np.ones((9, 19))
    mid = slice(2, 7)
    image[mid, :3] -= template[:, -3:]
    image[mid, 4:9] += template
    image[mid, -9:-4] -= template
    image[mid, -3:] += template[:, :3]
    result = match_template(image, template, pad_input=True, constant_values=image.mean())
    sorted_result = np.argsort(result.flat)
    i, j = np.unravel_index(sorted_result[:2], result.shape)
    assert_equal(j, (12, 0))
    i, j = np.unravel_index(sorted_result[-2:], result.shape)
    assert_equal(j, (18, 6))