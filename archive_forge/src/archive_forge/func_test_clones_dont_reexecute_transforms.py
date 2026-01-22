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
def test_clones_dont_reexecute_transforms():
    df = pd.DataFrame()
    msgs = []

    def piped(df, msg):
        msgs.append(msg)
        return df
    df.interactive.pipe(piped, msg='1').pipe(piped, msg='2')
    assert len(msgs) == 3