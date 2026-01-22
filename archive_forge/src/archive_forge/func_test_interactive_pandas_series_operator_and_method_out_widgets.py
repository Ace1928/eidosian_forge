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
def test_interactive_pandas_series_operator_and_method_out_widgets(series):
    w1 = pn.widgets.FloatSlider(value=2.0, start=1.0, end=5.0)
    w2 = pn.widgets.IntSlider(value=2, start=1, end=5)
    si = Interactive(series)
    si = (si + w1).head(w2)
    widgets = si.widgets()
    assert isinstance(widgets, pn.Column)
    assert len(widgets) == 2
    assert widgets[0] is w1
    assert widgets[1] is w2