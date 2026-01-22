from __future__ import annotations
import warnings
import numpy as np
import pytest
import xarray as xr
from xarray.tests import has_dask
@pytest.mark.parametrize('func', ['assert_equal', 'assert_identical', 'assert_allclose', 'assert_duckarray_equal', 'assert_duckarray_allclose'])
def test_ensure_warnings_not_elevated(func) -> None:

    class WarningVariable(xr.Variable):

        @property
        def dims(self):
            warnings.warn('warning in test')
            return super().dims

        def __array__(self):
            warnings.warn('warning in test')
            return super().__array__()
    a = WarningVariable('x', [1])
    b = WarningVariable('x', [2])
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('error')
        with pytest.raises(AssertionError):
            getattr(xr.testing, func)(a, b)
        assert len(w) > 0
        with pytest.raises(UserWarning):
            warnings.warn('test')
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('ignore')
        with pytest.raises(AssertionError):
            getattr(xr.testing, func)(a, b)
        assert len(w) == 0