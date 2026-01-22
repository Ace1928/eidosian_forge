import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def test_axis_matching_offset(self):
    fig = figure_grid([[{'data': [{'type': 'scatter', 'y': [1, 3, 2], 'xaxis': 'x', 'yaxis': 'y'}, {'type': 'bar', 'y': [2, 3, 1], 'xaxis': 'x2', 'yaxis': 'y2'}], 'layout': {'width': 400, 'height': 400, 'xaxis2': {'matches': 'x'}, 'yaxis2': {'matches': 'y'}}}], [{'data': [{'type': 'scatter', 'y': [1, 3, 2], 'xaxis': 'x', 'yaxis': 'y'}, {'type': 'bar', 'y': [2, 3, 1], 'xaxis': 'x2', 'yaxis': 'y2'}], 'layout': {'width': 400, 'height': 400, 'xaxis2': {'matches': 'y'}, 'yaxis2': {'matches': 'x'}}}]], column_spacing=0, row_spacing=0)
    self.assertEqual(fig['data'][0]['xaxis'], 'x')
    self.assertEqual(fig['data'][0]['yaxis'], 'y')
    self.assertEqual(fig['data'][1]['xaxis'], 'x2')
    self.assertEqual(fig['data'][1]['yaxis'], 'y2')
    self.assertEqual(fig['data'][2]['xaxis'], 'x3')
    self.assertEqual(fig['data'][2]['yaxis'], 'y3')
    self.assertEqual(fig['data'][3]['xaxis'], 'x4')
    self.assertEqual(fig['data'][3]['yaxis'], 'y4')
    self.assertEqual(fig['layout']['xaxis'].get('matches', None), None)
    self.assertEqual(fig['layout']['yaxis'].get('matches', None), None)
    self.assertEqual(fig['layout']['xaxis2'].get('matches', None), 'x')
    self.assertEqual(fig['layout']['yaxis2'].get('matches', None), 'y')
    self.assertEqual(fig['layout']['xaxis3'].get('matches', None), None)
    self.assertEqual(fig['layout']['yaxis3'].get('matches', None), None)
    self.assertEqual(fig['layout']['xaxis4'].get('matches', None), 'y3')
    self.assertEqual(fig['layout']['yaxis4'].get('matches', None), 'x3')