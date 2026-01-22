import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_element_polar_xlimits(self):
    theta = np.arange(0, 5.4, 0.1)
    r = np.ones(len(theta))
    scatter = Scatter((theta, r), 'theta', 'r').opts(projection='polar')
    plot = mpl_renderer.get_plot(scatter)
    ax = plot.handles['axis']
    self.assertIsInstance(ax, PolarAxes)
    self.assertEqual(ax.get_xlim(), (0, 2 * np.pi))