from breezy import tests
from breezy.revision import NULL_REVISION
from breezy.tests import per_workingtree
def test_pull_changes_root_id(self):
    tree = self.make_branch_and_tree('from')
    if not tree._format.supports_versioned_directories:
        self.skipTest('format does not support custom root ids')
    tree.set_root_id(b'first_root_id')
    self.build_tree(['from/file'])
    tree.add(['file'])
    tree.commit('first')
    to_tree = tree.controldir.sprout('to').open_workingtree()
    self.assertEqual(b'first_root_id', to_tree.path2id(''))
    tree.set_root_id(b'second_root_id')
    tree.commit('second')
    to_tree.pull(tree.branch)
    self.assertEqual(b'second_root_id', to_tree.path2id(''))