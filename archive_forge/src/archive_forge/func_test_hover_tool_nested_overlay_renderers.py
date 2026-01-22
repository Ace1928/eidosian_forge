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
def test_hover_tool_nested_overlay_renderers(self):
    overlay1 = NdOverlay({0: Curve(range(2)), 1: Curve(range(3))}, kdims=['Test'])
    overlay2 = NdOverlay({0: Curve(range(4)), 1: Curve(range(5))}, kdims=['Test'])
    nested_overlay = (overlay1 * overlay2).opts('Curve', tools=['hover'])
    plot = bokeh_renderer.get_plot(nested_overlay)
    self.assertEqual(len(plot.handles['hover'].renderers), 4)
    self.assertEqual(plot.handles['hover'].tooltips, [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])