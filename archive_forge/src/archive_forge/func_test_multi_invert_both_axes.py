from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_invert_both_axes(self):
    overlay = (Curve(range(10), vdims=['A']).opts(invert_yaxis=True) * Curve(range(10), vdims=['B']).opts(invert_yaxis=True)).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    y_range = plot.handles['y_range']
    self.assertEqual(y_range.start, 9)
    self.assertEqual(y_range.end, 0)
    extra_y_ranges = plot.handles['extra_y_ranges']
    self.assertEqual(list(extra_y_ranges.keys()), ['B'])
    self.assertEqual(extra_y_ranges['B'].start, 9)
    self.assertEqual(extra_y_ranges['B'].end, 0)