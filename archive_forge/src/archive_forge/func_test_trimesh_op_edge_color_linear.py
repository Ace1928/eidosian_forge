import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_trimesh_op_edge_color_linear(self):
    edges = [(0, 1, 2, 2.4), (1, 2, 3, 3.6)]
    nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
    trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
    plot = bokeh_renderer.get_plot(trimesh)
    cds = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    cmapper = plot.handles['edge_color_color_mapper']
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
    self.assertEqual(cds.data['edge_color'], np.array([2.4, 3.6]))
    self.assertEqual(cmapper.low, 2.4)
    self.assertEqual(cmapper.high, 3.6)