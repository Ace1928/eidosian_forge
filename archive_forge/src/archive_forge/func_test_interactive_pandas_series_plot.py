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
def test_interactive_pandas_series_plot(series, clone_spy):
    si = Interactive(series)
    si = si.plot()
    assert isinstance(si, Interactive)
    assert isinstance(si._current, matplotlib.axes.Axes)
    assert si._obj is series
    assert "dim('*').pd.plot(ax=<function Interactive._get_ax_fn.<locals>.get_ax" in repr(si._transform)
    assert si._depth == 3
    assert si._method is None
    assert clone_spy.count == 3
    assert clone_spy.calls[0].depth == 1
    assert not clone_spy.calls[0].args
    assert clone_spy.calls[0].kwargs == {'copy': True}
    assert clone_spy.calls[1].depth == 2
    assert not clone_spy.calls[1].args
    assert clone_spy.calls[1].kwargs == {'copy': True}
    assert clone_spy.calls[2].depth == 3
    assert len(clone_spy.calls[2].args) == 1
    assert "dim('*').pd.plot(ax=<function Interactive._get_ax_fn.<locals>.get_ax" in repr(clone_spy.calls[2].args[0])
    assert clone_spy.calls[2].kwargs == {'plot': True}
    assert not si._dmap
    assert isinstance(si._fig, matplotlib.figure.Figure)
    si.output()