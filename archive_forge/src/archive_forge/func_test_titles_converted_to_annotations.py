import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def test_titles_converted_to_annotations(self):
    fig = figure_grid([[{'data': [{'type': 'scatter', 'y': [1, 3, 2]}], 'layout': {'width': 400, 'height': 400, 'title': 'Scatter!'}}], [{'data': [{'type': 'bar', 'y': [2, 3, 1]}], 'layout': {'width': 400, 'height': 400, 'title': 'Bar!'}}]], row_spacing=100)
    go.Figure(fig)
    self.assertNotIn('title', fig['layout'])
    self.assertEqual(len(fig['layout']['annotations']), 2)
    self.assertEqual(fig['layout']['annotations'][0]['text'], 'Scatter!')
    self.assertEqual(fig['layout']['annotations'][1]['text'], 'Bar!')
    self.assertEqual(fig['layout']['width'], 400)
    self.assertEqual(fig['layout']['height'], 900)