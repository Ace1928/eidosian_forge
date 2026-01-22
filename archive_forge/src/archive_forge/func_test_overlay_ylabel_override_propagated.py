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
def test_overlay_ylabel_override_propagated(self):
    overlay = Curve(range(10)).opts(ylabel='custom y-label') * Curve(range(10))
    plot = bokeh_renderer.get_plot(overlay).state
    self.assertEqual(plot.yaxis[0].axis_label, 'custom y-label')