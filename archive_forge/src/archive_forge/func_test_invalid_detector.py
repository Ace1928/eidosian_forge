import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_invalid_detector(self):
    g = graph.DiGraph()
    g.add_node('a')
    g2 = graph.DiGraph()
    g2.add_node('c')
    self.assertRaises(ValueError, graph.merge_graphs, g, g2, overlap_detector='b')