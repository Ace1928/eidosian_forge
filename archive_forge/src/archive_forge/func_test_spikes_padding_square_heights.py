import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_padding_square_heights(self):
    spikes = Spikes([(1, 1), (2, 2), (3, 3)], vdims=['Height']).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(spikes)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.8)
    self.assertEqual(x_range.end, 3.2)
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 3.2)