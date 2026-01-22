from packaging.version import Version
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import matplotlib
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
from holoviews.util.transform import dim
from hvplot import bind
from hvplot.interactive import Interactive
from hvplot.tests.util import makeDataFrame, makeMixedDataFrame
from hvplot.xarray import XArrayInteractive
from hvplot.util import bokeh3, param2
@is_bokeh3
def test_interactive_pandas_layout_default_with_widgets_bk3(df):
    w = pn.widgets.IntSlider(value=2, start=1, end=5)
    dfi = Interactive(df)
    dfi = dfi.head(w)
    assert dfi._center is False
    assert dfi._loc == 'top_left'
    layout = dfi.layout()
    assert isinstance(layout, pn.Row)
    assert len(layout) == 1
    assert isinstance(layout[0], pn.Column)
    assert len(layout[0]) == 2
    assert isinstance(layout[0][0], pn.Column)
    assert isinstance(layout[0][1], pn.pane.PaneBase)
    assert len(layout[0][0]) == 1
    assert isinstance(layout[0][0][0], pn.widgets.IntSlider)