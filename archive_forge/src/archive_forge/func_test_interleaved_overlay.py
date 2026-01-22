import numpy as np
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve, Scatter
from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer
def test_interleaved_overlay(self):
    """
        Test to avoid regression after fix of https://github.com/holoviz/holoviews/issues/41
        """
    o = Overlay([Curve(np.array([[0, 1]])), Scatter([[1, 1]]), Curve(np.array([[0, 1]]))])
    OverlayPlot(o)