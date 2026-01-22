import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_node_count(self):
    root = self._make_species()
    self.assertEqual(7, 1 + root.child_count(only_direct=False))