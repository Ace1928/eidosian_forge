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
def test_overlay_muted_renderer(self):
    overlay = Curve(np.arange(5), label='increase') * Curve(np.arange(5) * -1 + 5, label='decrease').opts(muted=True)
    plot = bokeh_renderer.get_plot(overlay)
    unmuted, muted = plot.subplots.values()
    self.assertFalse(unmuted.handles['glyph_renderer'].muted)
    self.assertTrue(muted.handles['glyph_renderer'].muted)