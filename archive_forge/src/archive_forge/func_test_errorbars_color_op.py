import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import ErrorBars
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_errorbars_color_op(self):
    errorbars = ErrorBars([(0, 0, 0.1, 0.2, '#000'), (0, 1, 0.2, 0.4, '#F00'), (0, 2, 0.6, 1.2, '#0F0')], vdims=['y', 'perr', 'nerr', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(errorbars)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})