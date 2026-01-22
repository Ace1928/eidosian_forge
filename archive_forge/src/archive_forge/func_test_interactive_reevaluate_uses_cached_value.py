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
def test_interactive_reevaluate_uses_cached_value(series):
    w = pn.widgets.FloatSlider(value=2.0, start=1.0, end=5.0)
    si = Interactive(series)
    si = si + w
    w.value = 3.0
    assert repr(si._transform) == "dim('*').pd+FloatSlider(end=5.0, start=1.0, value=3.0)"
    assert si._callback().object is si._callback().object