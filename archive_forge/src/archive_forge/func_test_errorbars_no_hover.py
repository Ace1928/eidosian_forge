import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_errorbars_no_hover(self):
    plot = self.df_desc.hvplot.errorbars(y='mean', yerr1='std')
    assert list(plot.dimensions()) == ['index', 'mean', 'std']
    bkplot = Store.renderers['bokeh'].get_plot(plot)
    assert not bkplot.tools