from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def to_rgba_xr(image):
    data = image.data
    if cupy and isinstance(data, cupy.ndarray):
        data = cupy.asnumpy(data)
    shape = data.shape
    data = data.view(np.uint8).reshape(shape + (4,))
    return xr.DataArray(data, dims=image.dims + ('rgba',), coords=image.coords)