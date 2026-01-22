from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_graph_node_info_no_index(self):
    node_info = Dataset(np.arange(8), vdims=['Label'])
    graph = Graph(((self.source, self.target), node_info))
    self.assertEqual(graph.nodes.dimension_values(3), node_info.dimension_values(0))