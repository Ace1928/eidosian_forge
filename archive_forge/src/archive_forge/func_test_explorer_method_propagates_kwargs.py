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
def test_explorer_method_propagates_kwargs():
    explorer = df.hvplot.explorer(title='Dummy title', x='bill_length_mm')
    assert isinstance(explorer, hvDataFrameExplorer)
    assert explorer.kind == 'scatter'
    assert explorer.x == 'bill_length_mm'
    assert explorer.y == 'species'
    assert explorer.labels.title == 'Dummy title'