from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_graph_redim_nodes(self):
    graph = Graph(((self.target, self.source),))
    redimmed = graph.redim(x='x2', y='y2')
    self.assertEqual(redimmed.nodes, graph.nodes.redim(x='x2', y='y2'))
    self.assertEqual(redimmed.edgepaths, graph.edgepaths.redim(x='x2', y='y2'))