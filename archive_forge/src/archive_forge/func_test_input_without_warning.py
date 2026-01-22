import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
@parametrize('img', [np.ones((8, 8), dtype=bool), np.ones((4, 8, 8), dtype=bool)])
def test_input_without_warning(img):
    check_input(img)