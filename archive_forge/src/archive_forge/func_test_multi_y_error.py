import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_y_error(self):
    overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
    overlay.opts(multi_y=True)
    with pytest.raises(ValueError, match='multi_y and subcoordinate_y are not supported together'):
        bokeh_renderer.get_plot(overlay)