import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import HoloMap, NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Contours, Path, Polygons
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_polygons_categorical_color_op(self):
    polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'b'}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'a'}], vdims='color').opts(color='color')
    plot = bokeh_renderer.get_plot(polygons)
    cds = plot.handles['source']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    self.assertEqual(glyph.line_color, 'black')
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
    self.assertEqual(cds.data['color'], np.array(['b', 'a']))
    self.assertIsInstance(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['b', 'a'])