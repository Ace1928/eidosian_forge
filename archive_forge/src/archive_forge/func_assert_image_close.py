from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def assert_image_close(image0, image1, tolerance):

    def to_rgba_xr(image):
        data = image.data
        if cupy and isinstance(data, cupy.ndarray):
            data = cupy.asnumpy(data)
        shape = data.shape
        data = data.view(np.uint8).reshape(shape + (4,))
        return xr.DataArray(data, dims=image.dims + ('rgba',), coords=image.coords)
    da0 = to_rgba_xr(image0)
    da1 = to_rgba_xr(image1)
    xr.testing.assert_allclose(da0, da1, atol=tolerance, rtol=0)