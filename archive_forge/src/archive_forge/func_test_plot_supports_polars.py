from unittest import TestCase, SkipTest
import holoviews as hv
import pandas as pd
import pytest
from packaging.version import Version
from parameterized import parameterized
from hvplot.converter import HoloViewsConverter
from hvplot.plotting import plot
from hvplot.tests.util import makeDataFrame
def test_plot_supports_polars():
    pl = pytest.importorskip('polars')
    dfp = pl.DataFrame(makeDataFrame())
    out = plot(dfp, 'line')
    assert isinstance(out, hv.NdOverlay)
    assert out.keys() == dfp.columns