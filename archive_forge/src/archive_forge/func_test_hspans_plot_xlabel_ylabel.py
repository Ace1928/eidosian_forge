import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hspans_plot_xlabel_ylabel(self):
    hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra']).opts(xlabel='xlabel', ylabel='xlabel')
    plot = bokeh_renderer.get_plot(hspans)
    assert isinstance(plot.handles['glyph'], BkHStrip)
    assert plot.handles['xaxis'].axis_label == 'xlabel'
    assert plot.handles['yaxis'].axis_label == 'xlabel'