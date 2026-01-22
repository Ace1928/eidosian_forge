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
def test_path_colored_by_levels_single_value(self):
    xs = [1, 2, 3, 4]
    ys = xs[::-1]
    color = [998, 999, 998, 998]
    date = np.datetime64(dt.datetime(2018, 8, 1))
    data = {'x': xs, 'y': ys, 'color': color, 'date': date}
    levels = [0, 38, 73, 95, 110, 130, 156, 999]
    colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff6060']
    path = Path([data], vdims=['color', 'date']).opts(color_index='color', color_levels=levels, cmap=colors, tools=['hover'])
    plot = bokeh_renderer.get_plot(path)
    source = plot.handles['source']
    cmapper = plot.handles['color_mapper']
    self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
    self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
    self.assertEqual(source.data['color'], np.array([998, 999, 998]))
    self.assertEqual(source.data['date'], np.array([date] * 3))
    self.assertEqual(cmapper.low, 998)
    self.assertEqual(cmapper.high, 999)
    self.assertEqual(cmapper.palette, colors[-1:])