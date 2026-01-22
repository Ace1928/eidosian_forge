import numpy as np
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import Contours, Path, Polygons
from .test_plot import TestMPLPlot, mpl_renderer
def test_path_continuously_varying_line_width_op_update(self):
    xs = [1, 2, 3, 4]
    ys = xs[::-1]
    path = HoloMap({0: Path([{'x': xs, 'y': ys, 'line_width': [1, 7, 3, 2]}], vdims='line_width'), 1: Path([{'x': xs, 'y': ys, 'line_width': [3, 8, 2, 3]}], vdims='line_width')}).opts(linewidth='line_width')
    plot = mpl_renderer.get_plot(path)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_linewidths(), [1, 7, 3, 2])
    plot.update((1,))
    self.assertEqual(artist.get_linewidths(), [3, 8, 2, 3])