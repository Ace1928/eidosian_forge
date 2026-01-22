import numpy as np
from skimage._shared.testing import assert_almost_equal, assert_equal
from skimage import data, img_as_float
from skimage.morphology import diamond
from skimage.feature import match_template, peak_local_max
from skimage._shared import testing
def test_bounding_values():
    image = img_as_float(data.page())
    template = np.zeros((3, 3))
    template[1, 1] = 1
    result = match_template(image, template)
    print(result.max())
    assert result.max() < 1 + 1e-07
    assert result.min() > -1 - 1e-07