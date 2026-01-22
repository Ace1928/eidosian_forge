from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_build_commit_timestamp(self):
    """You can set a date when committing."""
    builder = self.make_branch_builder('foo')
    rev_id = builder.build_commit(timestamp=1236043340)
    branch = builder.get_branch()
    self.assertEqual((1, rev_id), branch.last_revision_info())
    rev = branch.repository.get_revision(branch.last_revision())
    self.assertEqual('commit 1', rev.message)
    self.assertEqual(1236043340, int(rev.timestamp))