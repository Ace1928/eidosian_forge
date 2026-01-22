import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_plot_graph_with_paths(self):
    graph = self.graph.clone((self.graph.data, self.graph.nodes, self.graph.edgepaths))
    plot = bokeh_renderer.get_plot(graph)
    node_source = plot.handles['scatter_1_source']
    edge_source = plot.handles['multi_line_1_source']
    layout_source = plot.handles['layout_source']
    self.assertEqual(node_source.data['index'], self.source)
    self.assertEqual(edge_source.data['start'], self.source)
    self.assertEqual(edge_source.data['end'], self.target)
    edges = graph.edgepaths.split()
    self.assertEqual(edge_source.data['xs'], [path.dimension_values(0) for path in edges])
    self.assertEqual(edge_source.data['ys'], [path.dimension_values(1) for path in edges])
    layout = {z: (x, y) for x, y, z in self.graph.nodes.array()}
    self.assertEqual(layout_source.graph_layout, layout)