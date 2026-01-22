from ....tests import TestCase, TestCaseWithTransport
from ....treebuilder import TreeBuilder
from ..maptree import MapTree, map_file_ids
def test_id2path(self):
    self.oldtree.lock_write()
    self.addCleanup(self.oldtree.unlock)
    builder = TreeBuilder()
    builder.start_tree(self.oldtree)
    builder.build(['foo'])
    builder.build(['bar'])
    builder.build(['bla'])
    builder.finish_tree()
    self.maptree = MapTree(self.oldtree, {})
    self.assertEqual('foo', self.maptree.id2path(self.maptree.path2id('foo')))