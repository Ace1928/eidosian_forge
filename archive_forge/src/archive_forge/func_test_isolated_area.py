import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_isolated_area():
    np.random.seed(0)
    a = np.random.random((7, 7))
    mask = -np.ones(a.shape)
    mask[1, 1] = 0
    mask[3:, 3:] = 0
    mask[4, 4] = 2
    mask[6, 6] = 1
    with expected_warnings(['The probability range is outside|scipy.sparse.linalg.cg']):
        res = random_walker(a, mask)
    assert res[1, 1] == 0
    with expected_warnings(['The probability range is outside|scipy.sparse.linalg.cg']):
        res = random_walker(a, mask, return_full_prob=True)
    assert res[0, 1, 1] == 0
    assert res[1, 1, 1] == 0