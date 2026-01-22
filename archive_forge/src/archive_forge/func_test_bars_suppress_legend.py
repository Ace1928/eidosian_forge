import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_suppress_legend(self):
    bars = Bars([('A', 1), ('B', 2)]).opts(show_legend=False)
    plot = bokeh_renderer.get_plot(bars)
    plot.initialize_plot()
    fig = plot.state
    self.assertEqual(len(fig.legend), 0)