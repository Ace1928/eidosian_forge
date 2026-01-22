import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def make_2d_syntheticdata(lx, ly=None):
    if ly is None:
        ly = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly)) + 0.1 * np.random.randn(lx, ly)
    small_l = int(lx // 5)
    data[lx // 2 - small_l:lx // 2 + small_l, ly // 2 - small_l:ly // 2 + small_l] = 1
    data[lx // 2 - small_l + 1:lx // 2 + small_l - 1, ly // 2 - small_l + 1:ly // 2 + small_l - 1] = 0.1 * np.random.randn(2 * small_l - 2, 2 * small_l - 2)
    data[lx // 2 - small_l, ly // 2 - small_l // 8:ly // 2 + small_l // 8] = 0
    seeds = np.zeros_like(data)
    seeds[lx // 5, ly // 5] = 1
    seeds[lx // 2 + small_l // 4, ly // 2 - small_l // 4] = 2
    return (data, seeds)