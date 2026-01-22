import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_disassociate_many(self):
    root = self._make_species()
    n = root.find('horse')
    n.parent.add(n)
    n.parent.add(n)
    c = n.disassociate()
    self.assertEqual(3, c)
    self.assertIsNone(n.parent)
    self.assertIsNone(root.find('horse'))