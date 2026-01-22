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
def test_batched_points_line_width_and_color(self):
    opts = {'NdOverlay': dict(legend_limit=0), 'Points': dict(line_width=Cycle(values=[0.5, 1]))}
    overlay = NdOverlay({i: Points([(i, j) for j in range(2)]) for i in range(2)}).opts(opts)
    plot = bokeh_renderer.get_plot(overlay).subplots[()]
    line_width = np.array([0.5, 0.5, 1.0, 1.0])
    color = np.array(['#30a2da', '#30a2da', '#fc4f30', '#fc4f30'], dtype='<U7')
    self.assertEqual(plot.handles['source'].data['line_width'], line_width)
    self.assertEqual(plot.handles['source'].data['color'], color)