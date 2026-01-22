import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_color_op(self):
    bars = Bars([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')], vdims=['y', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(bars)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color'})
    self.assertEqual(property_to_dict(glyph.line_color), 'black')