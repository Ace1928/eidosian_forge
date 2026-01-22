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
def test_dynamicmap_legend_updates(self):
    hmap = HoloMap({i: Curve([1, 2, 3], label=chr(65 + i + 2)) * Curve([1, 2, 3], label='B') for i in range(3)})
    dmap = Dynamic(hmap)
    plot = bokeh_renderer.get_plot(dmap)
    legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
    self.assertEqual(legend_labels, [{'value': 'C'}, {'value': 'B'}])
    plot.update((1,))
    legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
    self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'D'}])
    plot.update((2,))
    legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
    self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'E'}])