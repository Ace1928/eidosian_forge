import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Spikes
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_spikes_line_width_op_update(self):
    spikes = HoloMap({0: Spikes([(0, 0, 0.5), (0, 1, 3.2), (0, 2, 1.8)], vdims=['y', 'line_width']), 1: Spikes([(0, 0, 0.1), (0, 1, 0.8), (0, 2, 0.3)], vdims=['y', 'line_width'])}).opts(linewidth='line_width')
    plot = mpl_renderer.get_plot(spikes)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_linewidths(), [0.5, 3.2, 1.8])
    plot.update((1,))
    self.assertEqual(artist.get_linewidths(), [0.1, 0.8, 0.3])