from breezy import errors, tests
from breezy.tests.per_tree import TestCaseWithTree
def test_paths2ids_recursive(self):
    work_tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/dir/', 'tree/dir/file'])
    work_tree.add(['dir', 'dir/file'])
    if not work_tree.supports_setting_file_ids():
        raise tests.TestNotApplicable('test not applicable on non-inventory tests')
    tree = self._convert_tree(work_tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual({tree.path2id('dir'), tree.path2id('dir/file')}, tree.paths2ids(['dir']))