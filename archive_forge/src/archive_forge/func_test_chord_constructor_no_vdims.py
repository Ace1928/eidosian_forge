from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_chord_constructor_no_vdims(self):
    chord = Chord(self.simplices)
    nodes = np.array([[0.8660254037844387, 0.49999999999999994, 0], [-0.4999999999999998, 0.8660254037844388, 1], [-0.5000000000000004, -0.8660254037844384, 2], [0.8660254037844379, -0.5000000000000012, 3]])
    self.assertEqual(chord.nodes, Nodes(nodes))
    self.assertEqual(chord.array(), np.array([s[:2] for s in self.simplices]))