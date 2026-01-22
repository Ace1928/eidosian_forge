import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_multi_difference(self):
    s = sets.OrderedSet([1, 2, 3])
    s2 = sets.OrderedSet([2, 3])
    s3 = sets.OrderedSet([3, 4, 5])
    es = set(s)
    es2 = set(s2)
    es3 = set(s3)
    self.assertEqual(es3.difference(es), s3.difference(s))
    self.assertEqual(es.difference(es3), s.difference(s3))
    self.assertEqual(es2.difference(es, es3), s2.difference(s, s3))