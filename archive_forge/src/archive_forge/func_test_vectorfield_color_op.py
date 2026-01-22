import numpy as np
from holoviews.element import VectorField
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vectorfield_color_op(self):
    vectorfield = VectorField([(0, 0, 0, 1, '#000'), (0, 1, 0, 1, '#F00'), (0, 2, 0, 1, '#0F0')], vdims=['A', 'M', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(vectorfield)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0', '#000', '#F00', '#0F0', '#000', '#F00', '#0F0']))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})