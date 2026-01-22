import pandas as pd
from bokeh.models import FactorRange
from holoviews.core import NdOverlay
from holoviews.element import Segments
from .test_plot import TestBokehPlot, bokeh_renderer
def test_segments_color_selection_nonselection(self):
    opts = dict(color='green', selection_color='red', nonselection_color='blue')
    segments = Segments([(i, i * 2, i * 3, i * 4, i * 5, chr(65 + i)) for i in range(10)], vdims=['a', 'b']).opts(**opts)
    plot = bokeh_renderer.get_plot(segments)
    glyph_renderer = plot.handles['glyph_renderer']
    self.assertEqual(glyph_renderer.glyph.line_color, 'green')
    self.assertEqual(glyph_renderer.selection_glyph.line_color, 'red')
    self.assertEqual(glyph_renderer.nonselection_glyph.line_color, 'blue')