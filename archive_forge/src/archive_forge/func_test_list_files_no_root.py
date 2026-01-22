from breezy import osutils
from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
def test_list_files_no_root(self):
    work_tree = self.make_branch_and_tree('wt')
    tree = self.get_tree_no_parents_abc_content(work_tree)
    expected = [('a', 'V', 'file'), ('b', 'V', 'directory'), ('b/c', 'V', 'file')]
    self.assertFilesListEqual(tree, expected)