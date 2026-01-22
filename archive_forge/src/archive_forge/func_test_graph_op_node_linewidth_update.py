import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
def test_graph_op_node_linewidth_update(self):
    edges = [(0, 1), (0, 2)]

    def get_graph(i):
        c1, c2, c3 = {0: (2, 4, 6), 1: (12, 3, 5)}[i]
        nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)], vdims='line_width')
        return Graph((edges, nodes))
    graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_linewidth='line_width')
    plot = mpl_renderer.get_plot(graph)
    artist = plot.handles['nodes']
    self.assertEqual(artist.get_linewidths(), [2, 4, 6])
    plot.update((1,))
    self.assertEqual(artist.get_linewidths(), [12, 3, 5])