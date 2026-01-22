from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
def test_null_revid_revno(self):
    """null: should return revno 0."""
    branch = self.make_branch()
    self.assertEqual(0, branch.revision_id_to_revno(NULL_REVISION))