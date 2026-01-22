import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_removal_direct(self):
    root = self._make_species()
    self.assertRaises(ValueError, root.remove, 'human', only_direct=True)