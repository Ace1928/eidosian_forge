import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_small_large(self):
    for s in [5, 20, 51, 200, 1000]:
        d = np.random.randn(4, s)
        w = np.random.randint(0, d.size, size=d.size // 5)
        d.ravel()[w] = np.nan
        d[:, 0] = 1.0
        tgt = []
        for x in d:
            nonan = np.compress(~np.isnan(x), x)
            tgt.append(np.median(nonan, overwrite_input=True))
        assert_array_equal(np.nanmedian(d, axis=-1), tgt)