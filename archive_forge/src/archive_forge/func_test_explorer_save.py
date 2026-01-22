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
def test_explorer_save(tmp_path):
    explorer = hvplot.explorer(df)
    explorer.param.update(kind='scatter', x='bill_length_mm', y_multi=['bill_depth_mm'])
    outfile = tmp_path / 'plot.html'
    explorer.save(outfile)
    assert outfile.exists()