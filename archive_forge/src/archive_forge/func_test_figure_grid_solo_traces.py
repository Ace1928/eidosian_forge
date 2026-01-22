import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def test_figure_grid_solo_traces(self):
    fig = figure_grid([[{'data': [{'type': 'table', 'header': {'values': [['One', 'Two']]}}], 'layout': {'width': 400, 'height': 1000}}, {'data': [{'type': 'parcoords', 'dimensions': [{'values': [1, 2]}]}], 'layout': {'width': 600, 'height': 1000}}]], row_spacing=0, column_spacing=0)
    go.Figure(fig)
    self.assertEqual(fig['data'][0]['type'], 'table')
    self.assertEqual(fig['data'][0]['domain'], {'x': [0, 0.4], 'y': [0.0, 1.0]})
    self.assertEqual(fig['data'][1]['type'], 'parcoords')
    self.assertEqual(fig['data'][1]['domain'], {'x': [0.4, 1.0], 'y': [0, 1.0]})
    self.assertEqual(fig['layout']['width'], 1000)
    self.assertEqual(fig['layout']['height'], 1000)