import numpy as np
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.util import rgb2hex
from .test_plot import TestMPLPlot, mpl_renderer
def test_label_alpha_op_update(self):
    labels = HoloMap({0: Labels([(0, 0, 0.3), (0, 1, 1), (0, 2, 0.6)], vdims='alpha'), 1: Labels([(0, 0, 0.6), (0, 1, 0.1), (0, 2, 1)], vdims='alpha')}).opts(alpha='alpha')
    plot = mpl_renderer.get_plot(labels)
    artist = plot.handles['artist']
    self.assertEqual([a.get_alpha() for a in artist], [0.3, 1, 0.6])
    plot.update((1,))
    artist = plot.handles['artist']
    self.assertEqual([a.get_alpha() for a in artist], [0.6, 0.1, 1])