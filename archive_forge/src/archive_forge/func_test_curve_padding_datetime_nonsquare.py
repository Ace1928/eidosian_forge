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
def test_curve_padding_datetime_nonsquare(self):
    curve = Curve([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).opts(padding=0.1, width=600)
    plot = bokeh_renderer.get_plot(curve)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, np.datetime64('2016-03-31T21:36:00.000000000'))
    self.assertEqual(x_range.end, np.datetime64('2016-04-03T02:24:00.000000000'))
    self.assertEqual(y_range.start, 0.8)
    self.assertEqual(y_range.end, 3.2)