from breezy import branch, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.tests import per_branch
def test_update_local_commits_returns_old_tip(self):
    master_tree = self.make_branch_and_tree('master')
    child_tree = self.make_branch_and_tree('child')
    try:
        child_tree.branch.bind(master_tree.branch)
    except branch.BindingUnsupported:
        return
    child_tree.commit('foo', rev_id=b'foo', allow_pointless=True, local=True)
    master_tree.commit('bar', rev_id=b'bar', allow_pointless=True)
    self.assertEqual(b'foo', child_tree.branch.update())
    self.assertEqual(b'bar', child_tree.branch.last_revision())