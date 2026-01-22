import numpy as np
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.util import rgb2hex
from .test_plot import TestMPLPlot, mpl_renderer
def test_label_size_op(self):
    labels = Labels([(0, 0, 8), (0, 1, 12), (0, 2, 6)], vdims='size').opts(size='size')
    plot = mpl_renderer.get_plot(labels)
    artist = plot.handles['artist']
    self.assertEqual([a.get_fontsize() for a in artist], [8, 12, 6])