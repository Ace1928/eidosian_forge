import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_removal_self(self):
    root = self._make_species()
    n = root.find('horse')
    self.assertIsNotNone(n.parent)
    n.remove('horse', include_self=True)
    self.assertIsNone(n.parent)
    self.assertIsNone(root.find('horse'))