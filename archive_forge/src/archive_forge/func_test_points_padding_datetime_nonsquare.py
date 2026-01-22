import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_points_padding_datetime_nonsquare(self):
    points = Points([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).opts(padding=0.1, width=600)
    plot = bokeh_renderer.get_plot(points)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, np.datetime64('2016-03-31T21:36:00.000000000'))
    self.assertEqual(x_range.end, np.datetime64('2016-04-03T02:24:00.000000000'))
    self.assertEqual(y_range.start, 0.8)
    self.assertEqual(y_range.end, 3.2)