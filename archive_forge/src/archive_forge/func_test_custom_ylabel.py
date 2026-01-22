import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_custom_ylabel(self):
    overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
    overlay.opts(ylabel='Y label')
    plot = bokeh_renderer.get_plot(overlay)
    assert plot.state.yaxis.axis_label == 'Y label'
    assert plot.state.yaxis.ticker.ticks == [0, 1]
    assert plot.state.yaxis.major_label_overrides == {0: 'Data 0', 1: 'Data 1'}