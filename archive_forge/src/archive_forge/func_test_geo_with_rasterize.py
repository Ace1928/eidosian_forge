import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_geo_with_rasterize(self):
    import xarray as xr
    import cartopy.crs as ccrs
    import geoviews as gv
    try:
        from holoviews.operation.datashader import rasterize
    except:
        raise SkipTest('datashader not available')
    ds = xr.tutorial.open_dataset('air_temperature')
    hvplot_output = ds.isel(time=0).hvplot.points('lon', 'lat', crs=ccrs.PlateCarree(), projection=ccrs.LambertConformal(), rasterize=True, dynamic=False, aggregator='max', project=True)
    p1 = gv.Points(ds.isel(time=0), kdims=['lon', 'lat'], crs=ccrs.PlateCarree())
    p2 = gv.project(p1, projection=ccrs.LambertConformal())
    expected = rasterize(p2, dynamic=False, aggregator='max')
    xr.testing.assert_allclose(hvplot_output.data, expected.data)