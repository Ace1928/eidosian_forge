from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def check_eq_hist_cdf_slope(eq):
    cdf = np.histogram(eq[~np.isnan(eq)], bins=256)[0].cumsum()
    cdf = cdf / cdf[-1]
    slope = np.polyfit(np.linspace(0, 1, cdf.size), cdf, 1)[0]
    assert 0.9 < slope < 1.1