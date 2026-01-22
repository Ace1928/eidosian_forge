import numpy as np
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import Contours, Path, Polygons
from .test_plot import TestMPLPlot, mpl_renderer
def test_contours_line_width_op_update(self):
    contours = HoloMap({0: Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}], vdims='line_width'), 1: Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 2}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 5}], vdims='line_width')}).opts(linewidth='line_width', framewise=True)
    plot = mpl_renderer.get_plot(contours)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_linewidths(), [7, 3])
    plot.update((1,))
    self.assertEqual(artist.get_linewidths(), [2, 5])