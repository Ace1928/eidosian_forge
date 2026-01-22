import numpy as np
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels, Tiles
from .test_plot import TestPlotlyPlot
def test_labels_text_color_cycle(self):
    hm = HoloMap({i: Labels([(0, 0 + i, 'Label 1'), (1, 1 + i, 'Label 2')]) for i in range(3)}).overlay()
    assert isinstance(hm[0].opts['color'], Cycle)