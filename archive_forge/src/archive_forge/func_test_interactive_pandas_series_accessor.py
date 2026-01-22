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
def test_interactive_pandas_series_accessor(series, clone_spy):
    si = series.interactive()
    assert isinstance(si, Interactive)
    assert clone_spy.count == 1
    assert clone_spy.calls[0].is_empty()
    assert si._obj is series
    assert repr(si._transform) == "dim('*')"
    assert isinstance(si._current, pd.DataFrame)
    pd.testing.assert_series_equal(si._current.A, series)
    assert si._depth == 1
    assert si._method is None