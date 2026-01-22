from __future__ import annotations
import os
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.context import config
from numpy import nan
import datashader as ds
from datashader.datatypes import RaggedArray
import datashader.utils as du
import pytest
from datashader.tests.test_pandas import (
@pytest.mark.skipif(not sp, reason='spatialpandas not installed')
def test_points_geometry():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0.0, 2.0), 3), 3)
    ddf = dd.from_pandas(sp.GeoDataFrame({'geom': pd.array([[0, 0], [0, 1, 1, 1], [0, 2, 1, 2, 2, 2]], dtype='MultiPoint[float64]'), 'v': [1, 2, 3]}), npartitions=3)
    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(ddf, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan], [2, 2, nan], [3, 3, 3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)