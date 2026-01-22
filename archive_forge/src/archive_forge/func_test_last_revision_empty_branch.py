from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
def test_last_revision_empty_branch(self):
    branch = self.make_empty_branch()
    self.assertEqual(NULL_REVISION, branch.last_revision())
    self.assertEqual(0, branch.revno())
    self.assertEqual((0, NULL_REVISION), branch.last_revision_info())