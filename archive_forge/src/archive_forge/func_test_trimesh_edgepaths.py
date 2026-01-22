from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_trimesh_edgepaths(self):
    trimesh = TriMesh((self.simplices, self.nodes))
    paths = [np.array([(0, 0), (0.5, 1), (1, 0), (0, 0), (np.nan, np.nan), (0.5, 1), (1, 0), (1.5, 1), (0.5, 1)])]
    for p1, p2 in zip(trimesh.edgepaths.split(datatype='array'), paths):
        self.assertEqual(p1, p2)