import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_point_line_color_op(self):
    points = Points([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')], vdims='color').opts(line_color='color')
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['line_color'], np.array(['#000', '#F00', '#0F0']))
    self.assertNotEqual(property_to_dict(glyph.fill_color), {'field': 'line_color'})
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'line_color'})