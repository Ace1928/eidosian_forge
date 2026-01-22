import panel as pn
import param
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager
from holoviews import Curve, DynamicMap, HoloMap, Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly import PlotlyRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream
def test_render_holomap_scrubber(self):
    hmap = HoloMap({i: Curve([1, 2, i]) for i in range(5)})
    obj, _ = self.renderer._validate(hmap, 'scrubber')
    self.assertIsInstance(obj, pn.pane.HoloViews)
    self.assertEqual(obj.center, True)
    self.assertEqual(obj.widget_location, 'bottom')
    self.assertEqual(obj.widget_type, 'scrubber')
    widgets = obj.layout.select(Player)
    self.assertEqual(len(widgets), 1)
    player = widgets[0]
    self.assertEqual(player.start, 0)
    self.assertEqual(player.end, 4)