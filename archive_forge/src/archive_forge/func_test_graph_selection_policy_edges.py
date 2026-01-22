import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_selection_policy_edges(self):
    plot = bokeh_renderer.get_plot(self.graph.opts(selection_policy='edges'))
    renderer = plot.handles['glyph_renderer']
    hover = plot.handles['hover']
    self.assertIsInstance(renderer.selection_policy, EdgesAndLinkedNodes)
    self.assertIn(renderer, hover.renderers)