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
def test_polygon_with_hole_plot(self):
    xs = [1, 2, 3]
    ys = [2, 0, 7]
    holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
    poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
    plot = bokeh_renderer.get_plot(poly)
    source = plot.handles['source']
    self.assertEqual(source.data['xs'], [[[np.array([1, 2, 3, 1]), np.array([1.5, 2, 1.6, 1.5]), np.array([2.1, 2.5, 2.3, 2.1])]]])
    self.assertEqual(source.data['ys'], [[[np.array([2, 0, 7, 2]), np.array([2, 3, 1.6, 2]), np.array([4.5, 5, 3.5, 4.5])]]])