import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spikes_padding_logx(self):
    spikes = Spikes([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
    plot = bokeh_renderer.get_plot(spikes)
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.start, 0.8959584598407623)
    self.assertEqual(x_range.end, 3.348369522101713)