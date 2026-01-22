import numpy as np
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import Contours, Path, Polygons
from .test_plot import TestMPLPlot, mpl_renderer
def test_contours_categorical_color(self):
    path = Contours([{('x', 'y'): np.random.rand(10, 2), 'z': cat} for cat in ('B', 'A', 'B')], vdims='z').opts(color_index='z')
    plot = mpl_renderer.get_plot(path)
    artist = plot.handles['artist']
    self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 0]))
    self.assertEqual(artist.get_clim(), (0, 1))