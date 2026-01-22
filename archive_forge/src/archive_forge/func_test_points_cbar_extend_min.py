import matplotlib.pyplot as plt
import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Points
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_points_cbar_extend_min(self):
    img = Points(([0, 1], [0, 3])).redim(y=dict(range=(1, None)))
    plot = mpl_renderer.get_plot(img.opts(colorbar=True, color_index=1))
    self.assertEqual(plot.handles['cbar'].extend, 'min')