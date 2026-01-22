import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def strel_worker(self, fn, func):
    matlab_masks = np.load(fetch(fn))
    k = 0
    for arrname in sorted(matlab_masks):
        expected_mask = matlab_masks[arrname]
        actual_mask = func(k)
        if expected_mask.shape == (1,):
            expected_mask = expected_mask[:, np.newaxis]
        assert_equal(expected_mask, actual_mask)
        k = k + 1