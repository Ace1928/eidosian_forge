import unittest
from traits.observation._observer_graph import ObserverGraph
def test_eval_repr_roundtrip(self):
    graph = ObserverGraph(node=1, children=[ObserverGraph(node=2), ObserverGraph(node=3)])
    self.assertEqual(eval(repr(graph)), graph)