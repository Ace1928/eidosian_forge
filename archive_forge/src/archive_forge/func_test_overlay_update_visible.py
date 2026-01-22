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
def test_overlay_update_visible(self):
    hmap = HoloMap({i: Curve(np.arange(i), label='A') for i in range(1, 3)})
    hmap2 = HoloMap({i: Curve(np.arange(i), label='B') for i in range(3, 5)})
    plot = bokeh_renderer.get_plot(hmap * hmap2)
    subplot1, subplot2 = plot.subplots.values()
    self.assertTrue(subplot1.handles['glyph_renderer'].visible)
    self.assertFalse(subplot2.handles['glyph_renderer'].visible)
    plot.update((4,))
    self.assertFalse(subplot1.handles['glyph_renderer'].visible)
    self.assertTrue(subplot2.handles['glyph_renderer'].visible)