import numpy as np
from holoviews.core.spaces import DynamicMap
from holoviews.element import Spread
from holoviews.streams import Buffer
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spread_padding_soft_range(self):
    spread = Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.soft_range(y=(0, 3.5)).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(spread)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.8)
    self.assertEqual(x_range.end, 3.2)
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 3.5)