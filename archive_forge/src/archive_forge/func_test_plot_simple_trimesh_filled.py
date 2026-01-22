import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_plot_simple_trimesh_filled(self):
    plot = bokeh_renderer.get_plot(self.trimesh.opts(filled=True))
    node_source = plot.handles['scatter_1_source']
    edge_source = plot.handles['patches_1_source']
    layout_source = plot.handles['layout_source']
    self.assertIsInstance(plot.handles['patches_1_glyph'], Patches)
    self.assertEqual(node_source.data['index'], np.arange(4))
    self.assertEqual(edge_source.data['start'], np.arange(2))
    self.assertEqual(edge_source.data['end'], np.arange(1, 3))
    layout = {z: (x, y) for x, y, z in self.trimesh.nodes.array()}
    self.assertEqual(layout_source.graph_layout, layout)