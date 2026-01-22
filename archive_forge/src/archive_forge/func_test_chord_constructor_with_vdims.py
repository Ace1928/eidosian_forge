from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_chord_constructor_with_vdims(self):
    chord = Chord(self.simplices, vdims=['z'])
    nodes = np.array([[0.9396926207859084, 0.3420201433256687, 0], [6.123233995736766e-17, 1.0, 1], [-0.8660254037844388, -0.4999999999999998, 2], [0.7660444431189779, -0.6427876096865396, 3]])
    self.assertEqual(chord.nodes, Nodes(nodes))
    self.assertEqual(chord.array(), np.array(self.simplices))