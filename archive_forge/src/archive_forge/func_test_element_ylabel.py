import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_ylabel(self):
    element = Curve(range(10)).opts(ylabel='custom y-label')
    axes = mpl_renderer.get_plot(element).handles['axis']
    self.assertEqual(axes.get_ylabel(), 'custom y-label')