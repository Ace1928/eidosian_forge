import re
from textwrap import dedent
import holoviews as hv
import pandas as pd
import hvplot.pandas
import hvplot.xarray
import xarray as xr
import pytest
from bokeh.sampledata import penguins
from hvplot.ui import hvDataFrameExplorer, hvGridExplorer
def test_explorer_hvplot_gridded_two_variables():
    ds = ds_air_temperature.copy()
    ds['airx2'] = ds['air'] * 2
    explorer = hvplot.explorer(ds)
    assert isinstance(explorer, hvGridExplorer)
    assert isinstance(explorer._data, xr.DataArray)
    assert list(explorer._data['variable']) == ['air', 'airx2']
    assert explorer.kind == 'image'
    assert explorer.x == 'lat'
    assert explorer.y == 'lon'
    assert explorer.by == []
    assert explorer.groupby == ['time', 'variable']