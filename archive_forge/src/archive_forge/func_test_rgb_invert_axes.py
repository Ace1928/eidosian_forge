import numpy as np
import PIL.Image
import plotly.graph_objs as go
from holoviews.element import RGB, Tiles
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_rgb_invert_axes(self):
    rgb_data = np.random.rand(10, 10, 3)
    rgb = RGB(rgb_data).opts(invert_axes=True)
    fig_dict = plotly_renderer.get_plot_state(rgb)
    x_range = fig_dict['layout']['xaxis']['range']
    self.assertEqual(x_range[0], -0.5)
    self.assertEqual(x_range[1], 0.5)
    y_range = fig_dict['layout']['yaxis']['range']
    self.assertEqual(y_range[0], -0.5)
    self.assertEqual(y_range[1], 0.5)
    images = fig_dict['layout']['images']
    self.assertEqual(len(images), 1)
    image = images[0]
    self.assert_property_values(image, {'xref': 'x', 'yref': 'y', 'x': -0.5, 'y': 0.5, 'sizex': 1.0, 'sizey': 1.0, 'sizing': 'stretch', 'layer': 'above'})
    pil_img = self.rgb_element_to_pil_img(rgb.data)
    pil_img = pil_img.transpose(ROTATE_90).transpose(FLIP_LEFT_RIGHT)
    expected_source = go.layout.Image(source=pil_img).source
    self.assertEqual(image['source'], expected_source)