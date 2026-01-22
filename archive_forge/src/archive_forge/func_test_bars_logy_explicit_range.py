import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_logy_explicit_range(self):
    bars = Bars([('A', 1), ('B', 2), ('C', 3)], kdims=['Index'], vdims=['Value']).redim.range(Value=(0.001, 3))
    plot = bokeh_renderer.get_plot(bars.opts(logy=True))
    source = plot.handles['source']
    glyph = plot.handles['glyph']
    y_range = plot.handles['y_range']
    self.assertEqual(list(source.data['Index']), ['A', 'B', 'C'])
    self.assertEqual(source.data['Value'], np.array([1, 2, 3]))
    self.assertEqual(glyph.bottom, 0.001)
    self.assertEqual(y_range.start, 0.001)
    self.assertEqual(y_range.end, 3)