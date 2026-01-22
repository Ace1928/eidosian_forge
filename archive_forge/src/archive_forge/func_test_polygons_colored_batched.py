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
def test_polygons_colored_batched(self):
    polygons = NdOverlay({j: Polygons([[(i ** j, i, j) for i in range(10)]], vdims='Value') for j in range(5)}).opts(legend_limit=0)
    plot = next(iter(bokeh_renderer.get_plot(polygons).subplots.values()))
    cmapper = plot.handles['color_mapper']
    self.assertEqual(cmapper.low, 0)
    self.assertEqual(cmapper.high, 4)
    source = plot.handles['source']
    self.assertEqual(plot.handles['glyph'].fill_color['transform'], cmapper)
    self.assertEqual(source.data['Value'], list(range(5)))