from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def make_branch_with_revision_ids(self, *revision_ids):
    """Makes a branch with the given commits."""
    tree = self.make_branch_and_memory_tree('source')
    tree.lock_write()
    tree.add('')
    for revision_id in revision_ids:
        tree.commit('Message of ' + revision_id.decode('utf8'), rev_id=revision_id)
    tree.unlock()
    branch = tree.branch
    return branch