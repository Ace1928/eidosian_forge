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
def test_interactive_pandas_frame_filtering(df, clone_spy):
    dfi = Interactive(df)
    dfi = dfi[dfi.A > 1]
    assert isinstance(dfi, Interactive)
    assert dfi._obj is df
    assert isinstance(dfi._current, pd.DataFrame)
    pd.testing.assert_frame_equal(dfi._current, df[df.A > 1])
    assert repr(dfi._transform) == "dim('*', getitem, dim('*').pd.A)>1"
    assert dfi._depth == 2
    assert dfi._method is None
    assert clone_spy.count == 5
    assert clone_spy.calls[0].depth == 1
    assert not clone_spy.calls[0].args
    assert clone_spy.calls[0].kwargs == {'copy': True}
    assert clone_spy.calls[1].depth == 2
    assert len(clone_spy.calls[1].args) == 1
    assert repr(clone_spy.calls[1].args[0]) == "dim('*').pd.A()"
    assert clone_spy.calls[1].kwargs == {'inherit_kwargs': {}}
    assert clone_spy.calls[2].depth == 3
    assert len(clone_spy.calls[2].args) == 1
    assert repr(clone_spy.calls[2].args[0]) == "(dim('*').pd.A())>1"
    assert not clone_spy.calls[2].kwargs
    assert clone_spy.calls[3].depth == 1
    assert not clone_spy.calls[3].args
    assert clone_spy.calls[3].kwargs == {'copy': True}
    assert clone_spy.calls[4].depth == 2
    assert len(clone_spy.calls[4].args) == 1
    assert repr(clone_spy.calls[4].args[0]) == "dim('*', getitem, (dim('*').pd.A())>1)"
    assert not clone_spy.calls[4].kwargs