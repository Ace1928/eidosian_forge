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
def test_points_no_single_item_legend(self):
    points = Points([('A', 1), ('B', 2)], label='A')
    plot = bokeh_renderer.get_plot(points)
    plot.initialize_plot()
    fig = plot.state
    self.assertEqual(len(fig.legend), 0)