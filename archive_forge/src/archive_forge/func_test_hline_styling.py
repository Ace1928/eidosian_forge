import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_hline_styling(self):
    self.assert_shape_element_styling(HLine(3))