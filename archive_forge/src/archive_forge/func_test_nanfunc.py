import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
@pytest.mark.parametrize('nanfunc,func', nanfuncs.items(), ids=nanfunc_ids)
@np.errstate(over='ignore')
def test_nanfunc(self, mat, dtype, nanfunc, func):
    mat = mat.astype(dtype)
    tgt = func(mat)
    out = nanfunc(mat)
    assert_almost_equal(out, tgt)
    if dtype == 'O':
        assert type(out) is type(tgt)
    else:
        assert out.dtype == tgt.dtype