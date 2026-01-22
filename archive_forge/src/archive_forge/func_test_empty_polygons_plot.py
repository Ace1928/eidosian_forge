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
def test_empty_polygons_plot(self):
    poly = Polygons([], vdims=['Intensity'])
    plot = bokeh_renderer.get_plot(poly)
    source = plot.handles['source']
    self.assertEqual(len(source.data['xs']), 0)
    self.assertEqual(len(source.data['ys']), 0)
    self.assertEqual(len(source.data['Intensity']), 0)