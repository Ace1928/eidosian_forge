from .. import mutabletree, tests
def test_with_uncommitted_changes(self):
    self.build_tree(['tree/file'])
    self.tree.add('file')
    self.assertTrue(self.tree.has_changes())