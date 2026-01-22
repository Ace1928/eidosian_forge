from breezy import osutils
from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
def test_list_files_from_dir_no_recurse(self):
    work_tree = self.make_branch_and_tree('wt')
    tree = self.get_tree_no_parents_abc_content(work_tree)
    expected = [('a', 'V', 'file')]
    expected.append(('b', 'V', 'directory'))
    self.assertFilesListEqual(tree, expected, from_dir='', recursive=False)