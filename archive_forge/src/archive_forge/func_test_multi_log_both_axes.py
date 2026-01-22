from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_log_both_axes(self):
    overlay = (Curve(range(1, 9), vdims=['A']).opts(logy=True) * Curve(range(1, 9), vdims=['B']).opts(logy=True)).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    self.assertEqual(len(plot.state.yaxis), 2)
    self.assertTrue(isinstance(plot.state.yaxis[0], LogAxis))
    self.assertTrue(isinstance(plot.state.yaxis[1], LogAxis))
    extra_y_ranges = plot.handles['extra_y_scales']
    self.assertTrue(isinstance(extra_y_ranges['B'], LogScale))