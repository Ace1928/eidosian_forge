import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_arrow_plot_up(self):
    arrow = Arrow(0, 0, 'Test', '^')
    plot = bokeh_renderer.get_plot(arrow)
    self._compare_arrow_plot(plot, (0, -1 / 6.0), (0, 0))