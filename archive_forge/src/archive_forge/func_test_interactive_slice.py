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
@pytest.mark.skipif(Version(hv.__version__) < Version('1.15.1'), reason='Needs holoviews 1.15.1')
def test_interactive_slice():
    df = makeDataFrame()
    w = pn.widgets.IntSlider(start=10, end=40)
    idf = Interactive(df)
    pipeline = idf.iloc[:w]
    ioutput = pipeline.panel().object().object
    iw = pipeline.widgets()
    output = df.iloc[:10]
    pd.testing.assert_frame_equal(ioutput, output)
    assert len(iw) == 1
    assert iw[0] == w
    w.value = 15
    ioutput = pipeline.panel().object().object
    output = df.iloc[:15]
    pd.testing.assert_frame_equal(ioutput, output)