from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_trimesh_constructor_df_nodes(self):
    nodes_df = pd.DataFrame(self.nodes, columns=['x', 'y', 'z'])
    trimesh = TriMesh((self.simplices, nodes_df))
    nodes = Nodes([(0, 0, 0, 0), (0.5, 1, 1, 1), (1.0, 0, 2, 2), (1.5, 1, 3, 4)], vdims='z')
    self.assertEqual(trimesh.array(), np.array(self.simplices))
    self.assertEqual(trimesh.nodes, nodes)