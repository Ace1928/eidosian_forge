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
def test_points_datetime_hover(self):
    points = Points([(0, 1, dt.datetime(2017, 1, 1))], vdims='date').opts(tools=['hover'])
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['date'].astype('datetime64'), np.array([1483228800000000000]))
    hover = plot.handles['hover']
    self.assertEqual(hover.tooltips, [('x', '@{x}'), ('y', '@{y}'), ('date', '@{date}{%F %T}')], {'@{date}': 'datetime'})