from breezy import tests
from breezy.memorytree import MemoryTree
from breezy.tests import TestCaseWithTransport
from breezy.treebuilder import AlreadyBuilding, NotBuilding, TreeBuilder
def test_start_tree_when_started_fails(self):
    builder = TreeBuilder()
    tree = FakeTree()
    builder.start_tree(tree)
    self.assertRaises(AlreadyBuilding, builder.start_tree, tree)