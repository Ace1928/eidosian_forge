import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_no_xaxis_yaxis(self):
    element = Curve(range(10)).opts(xaxis=None, yaxis=None)
    axes = mpl_renderer.get_plot(element).handles['axis']
    xaxis = axes.get_xaxis()
    yaxis = axes.get_yaxis()
    self.assertEqual(xaxis.get_visible(), False)
    self.assertEqual(yaxis.get_visible(), False)