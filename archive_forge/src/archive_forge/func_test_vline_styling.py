import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_vline_styling(self):
    self.assert_shape_element_styling(VLine(3))