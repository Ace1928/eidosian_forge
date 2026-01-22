import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_multi_union(self):
    s = sets.OrderedSet([1, 2, 3])
    s2 = sets.OrderedSet([2, 3, 4])
    s3 = sets.OrderedSet([4, 5, 6])
    es = set(s)
    es2 = set(s2)
    es3 = set(s3)
    self.assertEqual(es.union(es2, es3), s.union(s2, s3))