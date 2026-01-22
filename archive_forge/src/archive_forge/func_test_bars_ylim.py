import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_ylim(self):
    bars = Bars([1, 2, 3]).opts(ylim=(0, 200))
    plot = bokeh_renderer.get_plot(bars)
    y_range = plot.handles['y_range']
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 200)