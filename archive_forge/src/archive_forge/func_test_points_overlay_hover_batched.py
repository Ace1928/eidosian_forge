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
def test_points_overlay_hover_batched(self):
    obj = NdOverlay({i: Points(np.random.rand(10, 2)) for i in range(5)}, kdims=['Test'])
    opts = {'Points': {'tools': ['hover']}, 'NdOverlay': {'legend_limit': 0}}
    obj = obj.opts(opts)
    self._test_hover_info(obj, [('Test', '@{Test}'), ('x', '@{x}'), ('y', '@{y}')])