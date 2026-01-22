import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_dfs_itr(self):
    root = self._make_species()
    things = list([n.item for n in root.dfs_iter(include_self=True)])
    self.assertEqual(set(['animal', 'reptile', 'mammal', 'horse', 'primate', 'monkey', 'human']), set(things))