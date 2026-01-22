import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_empty_spikes_plot(self):
    spikes = Spikes([], vdims=['Intensity'])
    plot = bokeh_renderer.get_plot(spikes)
    source = plot.handles['source']
    self.assertEqual(len(source.data['x']), 0)
    self.assertEqual(len(source.data['y0']), 0)
    self.assertEqual(len(source.data['y1']), 0)