from breezy import tests
from breezy.memorytree import MemoryTree
from breezy.tests import TestCaseWithTransport
from breezy.treebuilder import AlreadyBuilding, NotBuilding, TreeBuilder
def test_finish_tree_unlocks(self):
    builder = TreeBuilder()
    tree = FakeTree()
    builder.start_tree(tree)
    builder.finish_tree()
    self.assertEqual(['lock_tree_write', 'unlock'], tree._calls)