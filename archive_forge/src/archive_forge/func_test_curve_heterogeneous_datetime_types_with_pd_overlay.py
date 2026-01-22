import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models import FactorRange, FixedTicker
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.options import AbbreviatedException, Cycle, Palette
from holoviews.element import Curve
from holoviews.plotting.bokeh.callbacks import Callback, PointerXCallback
from holoviews.plotting.util import rgb2hex
from holoviews.streams import PointerX
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_curve_heterogeneous_datetime_types_with_pd_overlay(self):
    dates_pd = pd.date_range('2016-01-04', '2016-01-13', freq='D')
    dates64 = [np.datetime64(dt.datetime(2016, 1, i)) for i in range(1, 11)]
    dates = [dt.datetime(2016, 1, i) for i in range(2, 12)]
    curve_dt64 = Curve((dates64, np.random.rand(10)))
    curve_dt = Curve((dates, np.random.rand(10)))
    curve_pd = Curve((dates_pd, np.random.rand(10)))
    plot = bokeh_renderer.get_plot(curve_dt * curve_dt64 * curve_pd)
    self.assertEqual(plot.handles['x_range'].start, np.datetime64(dt.datetime(2016, 1, 1)))
    self.assertEqual(plot.handles['x_range'].end, np.datetime64(dt.datetime(2016, 1, 13)))