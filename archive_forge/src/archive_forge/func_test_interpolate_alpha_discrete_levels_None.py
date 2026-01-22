from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
@pytest.mark.parametrize('array_module', array_modules)
def test_interpolate_alpha_discrete_levels_None(array_module):
    data = array_module.array([[0.0, 1.0], [1.0, 0.0]])
    tf._interpolate_alpha(data, data, None, 'eq_hist', 0.5, None, 0.4, True)