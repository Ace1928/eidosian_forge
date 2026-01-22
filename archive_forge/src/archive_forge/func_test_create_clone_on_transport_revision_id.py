from breezy import branch, errors, tests
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.transport import FileExists, NoSuchFile
def test_create_clone_on_transport_revision_id(self):
    tree = self.make_branch_and_tree('source')
    old_revid = tree.commit('a commit')
    source_tip = tree.commit('a second commit')
    source = tree.branch
    target_transport = self.get_transport('target')
    result = tree.branch.create_clone_on_transport(target_transport, revision_id=old_revid)
    self.assertEqual(old_revid, result.last_revision())
    result.lock_read()
    self.addCleanup(result.unlock)
    self.assertFalse(result.repository.has_revision(source_tip))