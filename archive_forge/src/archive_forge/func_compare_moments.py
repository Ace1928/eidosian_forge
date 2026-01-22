import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
def compare_moments(m1, m2, thresh=1e-08):
    """Compare two moments arrays.

    Compares only values in the upper-left triangle of m1, m2 since
    values below the diagonal exceed the specified order and are not computed
    when the analytical computation is used.

    Also, there the first-order central moments will be exactly zero with the
    analytical calculation, but will not be zero due to limited floating point
    precision when using a numerical computation. Here we just specify the
    tolerance as a fraction of the maximum absolute value in the moments array.
    """
    m1 = m1.copy()
    m2 = m2.copy()
    nan_idx1 = np.where(np.isnan(m1.ravel()))[0]
    nan_idx2 = np.where(np.isnan(m2.ravel()))[0]
    assert len(nan_idx1) == len(nan_idx2)
    assert np.all(nan_idx1 == nan_idx2)
    m1[np.isnan(m1)] = 0
    m2[np.isnan(m2)] = 0
    max_val = np.abs(m1[m1 != 0]).max()
    for orders in itertools.product(*(range(m1.shape[0]),) * m1.ndim):
        if sum(orders) > m1.shape[0] - 1:
            m1[orders] = 0
            m2[orders] = 0
            continue
        abs_diff = abs(m1[orders] - m2[orders])
        rel_diff = abs_diff / max_val
        assert rel_diff < thresh