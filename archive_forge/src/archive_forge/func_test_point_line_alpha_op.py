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
def test_point_line_alpha_op(self):
    points = Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims='alpha').opts(line_alpha='alpha')
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
    self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'line_alpha'})
    self.assertNotEqual(property_to_dict(glyph.fill_alpha), {'field': 'line_alpha'})