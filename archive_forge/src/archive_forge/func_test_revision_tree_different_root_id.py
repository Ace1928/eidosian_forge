from breezy import errors, tests
from breezy import transport as _mod_transport
from breezy.tests import per_workingtree
def test_revision_tree_different_root_id(self):
    """A revision tree might have a very different root."""
    tree = self.make_branch_and_tree('tree1')
    if not tree.supports_setting_file_ids():
        raise tests.TestNotApplicable('tree does not support setting file ids')
    tree.set_root_id(b'one')
    rev1 = tree.commit('first post')
    tree.set_root_id(b'two')
    try:
        cached_revision_tree = tree.revision_tree(rev1)
    except errors.NoSuchRevision:
        return
    repository_revision_tree = tree.branch.repository.revision_tree(rev1)
    self.assertTreesEqual(repository_revision_tree, cached_revision_tree)