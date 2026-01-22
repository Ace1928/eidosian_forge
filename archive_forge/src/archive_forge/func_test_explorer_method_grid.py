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
def test_explorer_method_grid():
    explorer = ds_air_temperature.hvplot.explorer()
    assert isinstance(explorer, hvGridExplorer)
    assert explorer.kind == 'image'
    assert explorer.x == 'lat'
    assert explorer.y == 'lon'