import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_pydot_output(self):
    for graph_cls, kind, edge in [(graph.OrderedDiGraph, 'digraph', '->'), (graph.OrderedGraph, 'graph', '--')]:
        g = graph_cls(name='test')
        g.add_node('a')
        g.add_node('b')
        g.add_node('c')
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        expected = '\nstrict %(kind)s "test" {\na;\nb;\nc;\na %(edge)s b;\nb %(edge)s c;\n}\n' % {'kind': kind, 'edge': edge}
        self.assertEqual(expected.lstrip(), g.export_to_dot())