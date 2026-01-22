from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_y_disabled(self):
    overlay = Curve(range(10)) * Curve(range(10))
    plot = bokeh_renderer.get_plot(overlay).state
    self.assertEqual(len(plot.yaxis), 1)