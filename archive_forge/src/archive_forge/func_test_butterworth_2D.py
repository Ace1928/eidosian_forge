import math
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule
from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage.filters._fft_based import _get_nd_butterworth_filter
@pytest.mark.parametrize('high_pass', [True, False])
@pytest.mark.parametrize('squared_butterworth', [False, True])
def test_butterworth_2D(high_pass, squared_butterworth):
    order = 3 if squared_butterworth else 6
    im = np.random.randn(64, 128)
    filtered = butterworth(im, cutoff_frequency_ratio=0.2, order=order, high_pass=high_pass, squared_butterworth=squared_butterworth)
    im_fft = _fft_centered(im)
    im_fft = np.real(im_fft * np.conj(im_fft))
    filtered_fft = _fft_centered(filtered)
    filtered_fft = np.real(filtered_fft * np.conj(filtered_fft))
    outer_mask = np.ones(im.shape, dtype=bool)
    outer_mask[4:-4, 4:-4] = 0
    abs_filt_outer = filtered_fft[outer_mask].mean()
    abs_im_outer = im_fft[outer_mask].mean()
    inner_sl = tuple((slice(s // 2 - 4, s // 2 + 4) for s in im.shape))
    abs_filt_inner = filtered_fft[inner_sl].mean()
    abs_im_inner = im_fft[inner_sl].mean()
    if high_pass:
        assert abs_filt_outer > 0.9 * abs_im_outer
        assert abs_filt_inner < 0.1 * abs_im_inner
    else:
        assert abs_filt_outer < 0.1 * abs_im_outer
        assert abs_filt_inner > 0.9 * abs_im_inner