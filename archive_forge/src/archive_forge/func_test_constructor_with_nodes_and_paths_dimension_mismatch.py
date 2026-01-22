from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_constructor_with_nodes_and_paths_dimension_mismatch(self):
    paths = Graph(((self.source, self.target), self.nodes)).edgepaths
    exception = 'Ensure that the first two key dimensions on Nodes and EdgePaths match: x != x2'
    with self.assertRaisesRegex(ValueError, exception):
        Graph(((self.source, self.target), self.nodes, paths.redim(x='x2')))