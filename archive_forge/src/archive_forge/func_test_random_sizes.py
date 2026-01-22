import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
def test_random_sizes(self):
    elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    for m, n in np.random.randint(1, 101, size=(10, 2)):
        mask = np.ones((m, n), dtype=np.uint8)
        image8 = np.ones((m, n), dtype=np.uint8)
        out8 = np.empty_like(image8)
        rank.mean(image=image8, footprint=elem, mask=mask, out=out8, shift_x=0, shift_y=0)
        assert_equal(image8.shape, out8.shape)
        rank.mean(image=image8, footprint=elem, mask=mask, out=out8, shift_x=+1, shift_y=+1)
        assert_equal(image8.shape, out8.shape)
        rank.geometric_mean(image=image8, footprint=elem, mask=mask, out=out8, shift_x=0, shift_y=0)
        assert_equal(image8.shape, out8.shape)
        rank.geometric_mean(image=image8, footprint=elem, mask=mask, out=out8, shift_x=+1, shift_y=+1)
        assert_equal(image8.shape, out8.shape)
        image16 = np.ones((m, n), dtype=np.uint16)
        out16 = np.empty_like(image8, dtype=np.uint16)
        rank.mean(image=image16, footprint=elem, mask=mask, out=out16, shift_x=0, shift_y=0)
        assert_equal(image16.shape, out16.shape)
        rank.mean(image=image16, footprint=elem, mask=mask, out=out16, shift_x=+1, shift_y=+1)
        assert_equal(image16.shape, out16.shape)
        rank.geometric_mean(image=image16, footprint=elem, mask=mask, out=out16, shift_x=0, shift_y=0)
        assert_equal(image16.shape, out16.shape)
        rank.geometric_mean(image=image16, footprint=elem, mask=mask, out=out16, shift_x=+1, shift_y=+1)
        assert_equal(image16.shape, out16.shape)
        rank.mean_percentile(image=image16, mask=mask, out=out16, footprint=elem, shift_x=0, shift_y=0, p0=0.1, p1=0.9)
        assert_equal(image16.shape, out16.shape)
        rank.mean_percentile(image=image16, mask=mask, out=out16, footprint=elem, shift_x=+1, shift_y=+1, p0=0.1, p1=0.9)
        assert_equal(image16.shape, out16.shape)