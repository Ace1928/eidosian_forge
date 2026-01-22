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
def test_explorer_code_opts():
    da = ds_air_temperature['air']
    explorer = hvplot.explorer(da, x='lon', y='lat', kind='image', opts={'color_levels': 3})
    code = explorer.code
    assert code == dedent("        da.hvplot(\n            colorbar=True,\n            groupby=['time'],\n            kind='image',\n            x='lon',\n            y='lat',\n            legend='bottom_right',\n            widget_location='bottom',\n        ).opts(\n            color_levels=3,\n        )")
    assert explorer._code_pane.object == dedent("        ```python\n        da.hvplot(\n            colorbar=True,\n            groupby=['time'],\n            kind='image',\n            x='lon',\n            y='lat',\n            legend='bottom_right',\n            widget_location='bottom',\n        ).opts(\n            color_levels=3,\n        )\n        ```")