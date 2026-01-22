import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_slope_invert_axes(self):
    hspan = Slope(2, 10).opts(invert_axes=True)
    plot = bokeh_renderer.get_plot(hspan)
    slope = plot.handles['glyph']
    self.assertEqual(slope.gradient, 0.5)
    self.assertEqual(slope.y_intercept, -5)