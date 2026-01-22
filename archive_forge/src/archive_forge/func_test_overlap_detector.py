import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_overlap_detector(self):
    g = graph.DiGraph()
    g.add_node('a')
    g.add_node('b')
    g.add_edge('a', 'b')
    g2 = graph.DiGraph()
    g2.add_node('a')
    g2.add_node('d')
    g2.add_edge('a', 'd')
    self.assertRaises(ValueError, graph.merge_graphs, g, g2)

    def occurrence_detector(to_graph, from_graph):
        return sum((1 for node in from_graph.nodes if node in to_graph))
    self.assertRaises(ValueError, graph.merge_graphs, g, g2, overlap_detector=occurrence_detector)
    g3 = graph.merge_graphs(g, g2, allow_overlaps=True)
    self.assertEqual(3, len(g3))
    self.assertTrue(g3.has_edge('a', 'b'))
    self.assertTrue(g3.has_edge('a', 'd'))