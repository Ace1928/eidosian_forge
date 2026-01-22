import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_out_dtype_error(self):
    for f in self.nanfuncs:
        for dtype in [np.bool_, np.int_, np.object_]:
            out = np.empty(_ndat.shape[0], dtype=dtype)
            assert_raises(TypeError, f, _ndat, axis=1, out=out)