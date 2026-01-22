from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_multiple_with_file_ids(self):
    tree = self.make_branch_and_tree('.')
    if not tree.supports_setting_file_ids():
        self.skipTest('tree does not support setting file ids')
    self.build_tree(['a', 'b', 'dir/', 'dir/subdir/', 'dir/subdir/foo'])
    tree.add(['a', 'b', 'dir', 'dir/subdir', 'dir/subdir/foo'], ids=[b'a-id', b'b-id', b'dir-id', b'subdir-id', b'foo-id'])
    self.assertTreeLayout([('', tree.path2id('')), ('a', b'a-id'), ('b', b'b-id'), ('dir/', b'dir-id'), ('dir/subdir/', b'subdir-id'), ('dir/subdir/foo', b'foo-id')], tree)