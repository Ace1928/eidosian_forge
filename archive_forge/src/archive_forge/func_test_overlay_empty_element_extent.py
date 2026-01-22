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
def test_overlay_empty_element_extent(self):
    overlay = Curve([]).redim.range(x=(-10, 10)) * Points([]).redim.range(y=(-20, 20))
    plot = bokeh_renderer.get_plot(overlay)
    extents = plot.get_extents(overlay, {})
    self.assertEqual(extents, (-10, -20, 10, 20))