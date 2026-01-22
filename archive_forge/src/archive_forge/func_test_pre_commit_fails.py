from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
def test_pre_commit_fails(self):
    tree = self.make_branch_and_memory_tree('branch')
    with tree.lock_write():
        tree.add('')

        class PreCommitException(Exception):

            def __init__(self, revid):
                self.revid = revid

        def hook_func(local, master, old_revno, old_revid, new_revno, new_revid, tree_delta, future_tree):
            raise PreCommitException(new_revid)
        branch.Branch.hooks.install_named_hook('pre_commit', self.capture_pre_commit_hook, None)
        branch.Branch.hooks.install_named_hook('pre_commit', hook_func, None)
        revids = [None, None, None]
        err = self.assertRaises(PreCommitException, tree.commit, 'message')
        revids[0] = err.revid
        branch.Branch.hooks['pre_commit'] = []
        branch.Branch.hooks.install_named_hook('pre_commit', self.capture_pre_commit_hook, None)
        for i in range(1, 3):
            revids[i] = tree.commit('message')
        self.assertEqual([('pre_commit', 0, revision.NULL_REVISION, 1, revids[0], self.get_rootfull_delta(tree.branch.repository, revids[0])), ('pre_commit', 0, revision.NULL_REVISION, 1, revids[1], self.get_rootfull_delta(tree.branch.repository, revids[1])), ('pre_commit', 1, revids[1], 2, revids[2], self.get_rootfull_delta(tree.branch.repository, revids[2]))], self.hook_calls)