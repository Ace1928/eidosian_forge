import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_label_font_size_op_ints(self):
    labels = Labels([(0, 0, 10), (0, 1, 4), (0, 2, 8)], vdims='size').opts(text_font_size='size')
    plot = bokeh_renderer.get_plot(labels)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['text_font_size'], ['10pt', '4pt', '8pt'])
    self.assertEqual(property_to_dict(glyph.text_font_size), {'field': 'text_font_size'})