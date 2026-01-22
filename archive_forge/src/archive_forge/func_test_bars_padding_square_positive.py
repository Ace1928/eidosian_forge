import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_padding_square_positive(self):
    points = Bars([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(points)
    y_range = plot.handles['y_range']
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 3.2)