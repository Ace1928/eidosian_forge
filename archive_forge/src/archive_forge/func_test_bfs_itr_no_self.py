import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_bfs_itr_no_self(self):
    root = self._make_species()
    things = list([n.item for n in root.bfs_iter(include_self=False)])
    self.assertEqual(['reptile', 'mammal', 'primate', 'horse', 'human', 'monkey'], things)