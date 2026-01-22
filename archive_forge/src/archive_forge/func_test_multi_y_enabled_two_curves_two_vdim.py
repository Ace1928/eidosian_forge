from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_y_enabled_two_curves_two_vdim(self):
    overlay = (Curve(range(10), vdims=['A']) * Curve(range(10), vdims=['B'])).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    self.assertEqual(len(plot.state.yaxis), 2)
    y_range = plot.handles['y_range']
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 9)
    extra_y_ranges = plot.handles['extra_y_ranges']
    self.assertEqual(list(extra_y_ranges.keys()), ['B'])
    self.assertEqual(extra_y_ranges['B'].start, 0)
    self.assertEqual(extra_y_ranges['B'].end, 9)