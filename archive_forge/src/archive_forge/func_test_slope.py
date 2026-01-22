import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_slope(self):
    hspan = Slope(2, 10)
    plot = bokeh_renderer.get_plot(hspan)
    slope = plot.handles['glyph']
    self.assertEqual(slope.gradient, 2)
    self.assertEqual(slope.y_intercept, 10)