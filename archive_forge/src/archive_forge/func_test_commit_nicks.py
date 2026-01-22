from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
def test_commit_nicks(self):
    """Nicknames are committed to the revision"""
    self.get_transport().mkdir('bzr.dev')
    wt = self.make_branch_and_tree('bzr.dev')
    branch = wt.branch
    branch.nick = 'My happy branch'
    wt.commit('My commit respect da nick.')
    committed = branch.repository.get_revision(branch.last_revision())
    if branch.repository._format.supports_storing_branch_nick:
        self.assertEqual(committed.properties['branch-nick'], 'My happy branch')
    else:
        self.assertNotIn('branch-nick', committed.properties)