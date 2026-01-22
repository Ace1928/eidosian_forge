from breezy import branch, errors, tests
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.transport import FileExists, NoSuchFile
def test_create_clone_on_transport_stacked(self):
    tree = self.make_branch_and_tree('source')
    tree.commit('a commit')
    trunk = tree.branch.create_clone_on_transport(self.get_transport('trunk'))
    revid = tree.commit('a second commit')
    source = tree.branch
    target_transport = self.get_transport('target')
    try:
        result = tree.branch.create_clone_on_transport(target_transport, stacked_on=trunk.base)
    except branch.UnstackableBranchFormat:
        if not trunk.repository._format.supports_full_versioned_files:
            raise tests.TestNotApplicable('can not stack on format')
        raise
    self.assertEqual(revid, result.last_revision())
    self.assertEqual(trunk.base, result.get_stacked_on_url())