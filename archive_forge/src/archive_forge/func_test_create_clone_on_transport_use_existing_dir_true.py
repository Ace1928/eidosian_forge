from breezy import branch, errors, tests
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.transport import FileExists, NoSuchFile
def test_create_clone_on_transport_use_existing_dir_true(self):
    tree = self.make_branch_and_tree('source')
    tree.commit('a commit')
    source = tree.branch
    target_transport = self.get_transport('target')
    target_transport.create_prefix()
    result = tree.branch.create_clone_on_transport(target_transport, use_existing_dir=True)
    self.assertEqual(source.last_revision(), result.last_revision())