import unittest
from traits.observation._observer_graph import ObserverGraph
def test_equality_different_length_children(self):
    graph1 = ObserverGraph(node=1, children=[ObserverGraph(node=2), ObserverGraph(node=3)])
    graph2 = ObserverGraph(node=1, children=[ObserverGraph(node=2)])
    self.assertNotEqual(graph1, graph2)