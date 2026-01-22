import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import VectorField
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_vectorfield_color_op_update(self):
    vectorfield = HoloMap({0: VectorField([(0, 0, 0, 1, '#000000'), (0, 1, 0, 1, '#FF0000'), (0, 2, 0, 1, '#00FF00')], vdims=['A', 'M', 'color']), 1: VectorField([(0, 0, 0, 1, '#0000FF'), (0, 1, 0, 1, '#00FF00'), (0, 2, 0, 1, '#FF0000')], vdims=['A', 'M', 'color'])}).opts(color='color')
    plot = mpl_renderer.get_plot(vectorfield)
    artist = plot.handles['artist']
    self.assertEqual(artist.get_facecolors(), np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]))
    plot.update((1,))
    self.assertEqual(artist.get_facecolors(), np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]]))