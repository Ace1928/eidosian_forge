from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_all_file_ids(self):
    work_tree = self.make_branch_and_tree('wt')
    tree = self.get_tree_no_parents_abc_content(work_tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(tree.all_file_ids(), {tree.path2id('a'), tree.path2id(''), tree.path2id('b'), tree.path2id('b/c')})