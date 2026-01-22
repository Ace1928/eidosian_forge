import numpy as np
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.util import rgb2hex
from .test_plot import TestMPLPlot, mpl_renderer
def test_label_size_op_update(self):
    labels = HoloMap({0: Labels([(0, 0, 8), (0, 1, 6), (0, 2, 12)], vdims='size'), 1: Labels([(0, 0, 9), (0, 1, 4), (0, 2, 3)], vdims='size')}).opts(size='size')
    plot = mpl_renderer.get_plot(labels)
    artist = plot.handles['artist']
    self.assertEqual([a.get_fontsize() for a in artist], [8, 6, 12])
    plot.update((1,))
    artist = plot.handles['artist']
    self.assertEqual([a.get_fontsize() for a in artist], [9, 4, 3])