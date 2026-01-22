import numpy as np
import panel as pn
from bokeh.models import FactorRange, FixedTicker, HoverTool, Range1d, Span
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import (
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream, Tap
from holoviews.util import Dynamic
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_overlay_framewise_norm(self):
    a = {'X': [0, 1, 2], 'Y': [0, 1, 2], 'Z': [0, 50, 100]}
    b = {'X': [3, 4, 5], 'Y': [0, 10, 20], 'Z': [50, 50, 150]}
    sa = Scatter(a, 'X', ['Y', 'Z']).opts(color='Z', framewise=True)
    sb = Scatter(b, 'X', ['Y', 'Z']).opts(color='Z', framewise=True)
    plot = bokeh_renderer.get_plot(sa * sb)
    sa_plot, sb_plot = plot.subplots.values()
    sa_cmapper = sa_plot.handles['color_color_mapper']
    sb_cmapper = sb_plot.handles['color_color_mapper']
    self.assertEqual(sa_cmapper.low, 0)
    self.assertEqual(sb_cmapper.low, 0)
    self.assertEqual(sa_cmapper.high, 150)
    self.assertEqual(sb_cmapper.high, 150)