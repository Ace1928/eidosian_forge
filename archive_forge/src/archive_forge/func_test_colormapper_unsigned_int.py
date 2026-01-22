import numpy as np
from matplotlib import style
from matplotlib.projections import PolarAxes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, PercentFormatter
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HeatMap, Image, Scatter, Scatter3D
from holoviews.streams import Stream
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_colormapper_unsigned_int(self):
    img = Image(np.array([[1, 1, 1, 2], [2, 2, 3, 4]]).astype('uint16'))
    plot = mpl_renderer.get_plot(img)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_clim(), (1, 4))