from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_existing_id(self):
    """Adding an entry with a pre-existing id raises DuplicateFileId"""
    tree = self.make_branch_and_tree('.')
    if not tree.supports_setting_file_ids():
        self.skipTest('tree does not support setting file ids')
    self.build_tree(['a', 'b'])
    tree.add(['a'])
    self.assertRaises(inventory.DuplicateFileId, tree.add, ['b'], ids=[tree.path2id('a')])
    self.assertTreeLayout(['', 'a'], tree)