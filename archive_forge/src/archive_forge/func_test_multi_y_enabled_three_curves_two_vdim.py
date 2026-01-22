from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_y_enabled_three_curves_two_vdim(self):
    curve_1A = Curve(range(10), vdims=['A'])
    curve_2B = Curve(range(11), vdims=['B'])
    curve_3A = Curve(range(12), vdims=['A'])
    overlay = (curve_1A * curve_2B * curve_3A).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay).state
    self.assertEqual(len(plot.yaxis), 2)