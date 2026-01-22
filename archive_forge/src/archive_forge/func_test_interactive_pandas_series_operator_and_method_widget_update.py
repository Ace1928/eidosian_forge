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
def test_interactive_pandas_series_operator_and_method_widget_update(series):
    w1 = pn.widgets.FloatSlider(value=2.0, start=1.0, end=5.0)
    w2 = pn.widgets.IntSlider(value=2, start=1, end=5)
    si = Interactive(series)
    si = (si + w1).head(w2)
    w1.value = 3.0
    w2.value = 3
    assert repr(si._transform) == "(dim('*').pd+FloatSlider(end=5.0, start=1.0, value=3.0)).head(IntSlider(end=5, start=1, value=3))"
    out = si._callback()
    assert out.object is si.eval()
    assert isinstance(out, pn.pane.DataFrame)
    pd.testing.assert_series_equal(out.object.A, (series + 3.0).head(3))