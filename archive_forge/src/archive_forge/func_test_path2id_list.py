from breezy import errors, tests
from breezy.tests.per_tree import TestCaseWithTree
def test_path2id_list(self):
    self.assertEqual(b'bla-id', self.tree_a.path2id(['bla']))
    self.assertEqual(b'dir-id', self.tree_a.path2id(['dir']))
    self.assertEqual(b'file-id', self.tree_a.path2id(['dir', 'file']))
    self.assertEqual(self.tree_a.path2id(''), self.tree_a.path2id([]))
    self.assertIs(None, self.tree_a.path2id(['idontexist']))
    self.assertIs(None, self.tree_a.path2id(['dir', 'idontexist']))