import panel as pn
import param
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager
from holoviews import Curve, DynamicMap, HoloMap, Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly import PlotlyRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream
def test_render_dynamicmap_with_dims(self):
    dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y']).redim.range(y=(0.1, 5))
    obj, _ = self.renderer._validate(dmap, None)
    self.renderer.components(obj)
    [(plot, pane)] = obj._plots.values()
    y = plot.handles['fig']['data'][0]['y']
    self.assertEqual(y[2], 0.1)
    slider = obj.layout.select(FloatSlider)[0]
    slider.value = 3.1
    y = plot.handles['fig']['data'][0]['y']
    self.assertEqual(y[2], 3.1)