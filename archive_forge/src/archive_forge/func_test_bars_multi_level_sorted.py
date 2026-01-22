import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_multi_level_sorted(self):
    box = Bars((['A', 'B'] * 15, [3, 10, 1] * 10, np.random.randn(30)), ['Group', 'Category'], 'Value').aggregate(function=np.mean)
    plot = bokeh_renderer.get_plot(box)
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.factors, [('A', '1'), ('A', '3'), ('A', '10'), ('B', '1'), ('B', '3'), ('B', '10')])