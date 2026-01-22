import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_op_node_line_width(self):
    edges = [(0, 1), (0, 2)]
    nodes = Nodes([(0, 0, 0, 2), (0, 1, 1, 4), (1, 1, 2, 6)], vdims='line_width')
    graph = Graph((edges, nodes)).opts(node_line_width='line_width')
    plot = bokeh_renderer.get_plot(graph)
    cds = plot.handles['scatter_1_source']
    glyph = plot.handles['scatter_1_glyph']
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'node_line_width'})
    self.assertEqual(cds.data['node_line_width'], np.array([2, 4, 6]))