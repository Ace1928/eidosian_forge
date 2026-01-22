from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_absent_directory(self):
    """Removing a absent directory succeeds without corruption (#150438)."""
    paths = ['a/', 'a/b']
    tree = self.get_committed_tree(paths)
    tree.controldir.root_transport.delete_tree('a')
    tree.remove(['a'])
    self.assertRemovedAndDeleted('b')
    tree._validate()