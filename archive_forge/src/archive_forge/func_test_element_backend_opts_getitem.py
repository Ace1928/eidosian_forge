import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_backend_opts_getitem(self):
    a = Curve([1, 2, 3], label='a')
    b = Curve([1, 4, 9], label='b')
    c = Curve([1, 4, 18], label='c')
    d = Curve([1, 4, 36], label='d')
    e = Curve([1, 4, 36], label='e')
    curve = (a * b * c * d * e).opts(show_legend=True, backend_opts={'legend.get_texts()[0].fontsize': 188, 'legend.get_texts()[1:3].fontsize': 288, 'legend.get_texts()[3,4].fontsize': 388})
    plot = mpl_renderer.get_plot(curve)
    legend = plot.handles['legend']
    self.assertEqual(legend.get_texts()[0].get_fontsize(), 188)
    self.assertEqual(legend.get_texts()[1].get_fontsize(), 288)
    self.assertEqual(legend.get_texts()[2].get_fontsize(), 288)
    self.assertEqual(legend.get_texts()[3].get_fontsize(), 388)
    self.assertEqual(legend.get_texts()[4].get_fontsize(), 388)